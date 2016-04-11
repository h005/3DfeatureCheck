#ifndef EXTERNALIMPORTER_H
#define EXTERNALIMPORTER_H

#include "common.hh"
#include <vector>
#include <utility>
#include <map>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/string_cast.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <assimp/DefaultLogger.hpp>
#include <assimp/LogStream.hpp>

template <typename MeshT>
class ExternalImporter
{
public:
    /**
     * @brief 读取一个外部文件，类似于OpenMesh::IO::read_mesh。 注意，得到的坐标已经进行了移中和缩放。
     * @param mesh 结果存放位置
     * @param path 文件路径
     * @param autoScaleAndShift 载入OpenMesh之前，是否进行移中和缩放
     * @return 是否读取成功
     * @see OpenMesh::IO::read_mesh
     */
    static bool read_mesh(MeshT &mesh, const char *path, bool autoScaleAndShift=false)
    {
        Assimp::Importer importer;
        const aiScene *scene = importer.ReadFile(path, aiProcessPreset_TargetRealtime_Quality);
        if (!scene)
            return false;

        glm::mat4 initTransform = glm::mat4();
        if (autoScaleAndShift) {
            // 得到移中和缩放变换，直接送给
            auto result = recommandScaleAndShift(scene);
            initTransform = glm::scale(glm::mat4(1.f), glm::vec3(result.first)) * result.second;
        }

        // 此时文件读取完毕，需要将Assimp中的数据结构转换为OpenMesh中的数据结构
        // aiScene呈现树形结构，递归转换每个节点
        int count = 0;
        std::map<std::pair<typename MeshT::VertexHandle, typename MeshT::VertexHandle>, typename MeshT::FaceHandle> halfEdgeLink; // 用于过滤非法的面
        recursive_create(scene, scene->mRootNode, initTransform, mesh, count, halfEdgeLink);

        std::cout << "Assimp Importer: " << count << " Meshes Loaded." << std::endl;
        return true;
    }

private:
    static void recursive_create(const aiScene *sc, const aiNode* nd, const glm::mat4 &inheritedTransformation, MeshT &openMesh, int &count,
                                 std::map<std::pair<typename MeshT::VertexHandle, typename MeshT::VertexHandle>, typename MeshT::FaceHandle> &halfEdgeLink)
    {
        assert(nd && sc);
        unsigned int n = 0;

        // 相对于父节点的变换，aiMatrix4x4中是行优先存储的，
        // 所以需要先将内存结构变为列优先存储
        glm::mat4 mTransformation = glm::transpose(glm::make_mat4((float *)&nd->mTransformation));
        glm::mat4 absoluteTransformation = inheritedTransformation * mTransformation;
        // 设置shader中的model view矩阵

        // 累计遍历的mesh个数，读取完毕后用于输出
        count += nd->mNumMeshes;
        //
        for (; n < nd->mNumMeshes; ++n)
        {
            // 一个aiNode中存有其mesh的索引，
            // 在aiScene中可以用这个索引拿到真正的aiMesh
            const struct aiMesh* mesh = sc->mMeshes[nd->mMeshes[n]];

            // 将所有点变换后，加入OpenMesh结构中，并保存它们的索引
            std::vector<typename MeshT::VertexHandle> vHandle;
            if(mesh->HasPositions()) {
                for(uint32_t i = 0; i < mesh->mNumVertices; ++i) {
                    glm::vec3 position(mesh->mVertices[i].x, mesh->mVertices[i].y, mesh->mVertices[i].z);
                    glm::vec4 absolutePosition = absoluteTransformation * glm::vec4(position, 1.f);

                    typename MeshT::Point point(absolutePosition.x, absolutePosition.y, absolutePosition.z);
                    vHandle.push_back(openMesh.add_vertex(point));
                }
            }

            if(mesh->HasFaces() && vHandle.size() > 0) {
                std::vector<typename MeshT::VertexHandle> fHandle(3);
                // 一个face代表一个面（暂时只考虑三角形，其余类型pass），其存储着各个顶点的索引
                // 可以根据索引到mesh->mVertices[]中找到对应顶点的数据(x, y, z)
                for(uint32_t i = 0; i < mesh->mNumFaces; ++i) {
                    if (mesh->mFaces[i].mNumIndices == 3) {
                        fHandle[0] = vHandle[mesh->mFaces[i].mIndices[0]];
                        fHandle[1] = vHandle[mesh->mFaces[i].mIndices[1]];
                        fHandle[2] = vHandle[mesh->mFaces[i].mIndices[2]];


                        // complex edge修正算法
                        if (halfEdgeLink.count(std::make_pair(fHandle[0], fHandle[1]))
                                || halfEdgeLink.count(std::make_pair(fHandle[1], fHandle[2]))
                                || halfEdgeLink.count(std::make_pair(fHandle[2], fHandle[0]))) {
                            // 如果半边已经存在，则反序
                            fHandle[2] = vHandle[mesh->mFaces[i].mIndices[0]];
                            fHandle[1] = vHandle[mesh->mFaces[i].mIndices[1]];
                            fHandle[0] = vHandle[mesh->mFaces[i].mIndices[2]];
                        }
                        if (halfEdgeLink.count(std::make_pair(fHandle[0], fHandle[1]))
                                || halfEdgeLink.count(std::make_pair(fHandle[1], fHandle[2]))
                                || halfEdgeLink.count(std::make_pair(fHandle[2], fHandle[0]))) {
                            // 反序之后还是冲突，则忽略该边
                            std::cout << "Drop face: " << fHandle[0] << " " << fHandle[1] << " " << fHandle[2] << std::endl;
                        } else {
                            typename MeshT::FaceHandle fh = openMesh.add_face(fHandle);

                            // 占有该半边
                            {
                                auto key = std::make_pair(fHandle[0], fHandle[1]);
                                assert(halfEdgeLink.count(key) == 0);
                                halfEdgeLink[key] = fh;
                            }
                            {
                                auto key = std::make_pair(fHandle[1], fHandle[2]);
                                assert(halfEdgeLink.count(key) == 0);
                                halfEdgeLink[key] = fh;
                            }
                            {
                                auto key = std::make_pair(fHandle[2], fHandle[0]);
                                assert(halfEdgeLink.count(key) == 0);
                                halfEdgeLink[key] = fh;
                            }
                        }


                    }
                }
            }
        }
        // create all children
        for (n = 0; n < nd->mNumChildren; ++n)
            recursive_create(sc, nd->mChildren[n], absoluteTransformation, openMesh, count, halfEdgeLink);
    }

    static std::pair<GLfloat, glm::mat4> recommandScaleAndShift(const aiScene *sc)
    {
        aiVector3D scene_min, scene_max, scene_center;
        get_bounding_box(sc, &scene_min, &scene_max);

        scene_center.x = (scene_min.x + scene_max.x) / 2.f;
        scene_center.y = (scene_min.y + scene_max.y) / 2.f;
        scene_center.z = (scene_min.z + scene_max.z) / 2.f;

        float tmp = -1e10;
        tmp = std::max(scene_max.x - scene_min.x, tmp);
        tmp = std::max(scene_max.y - scene_min.y, tmp);
        tmp = std::max(scene_max.z - scene_min.z, tmp);
        float scale = 2.f / tmp;
        glm::mat4 shiftTransform = glm::translate(glm::mat4(1.f), glm::vec3(-scene_center.x, -scene_center.y, -scene_center.z));
        return std::make_pair(scale, shiftTransform);
    }

    static void get_bounding_box_for_node(const aiScene *sc,
        const aiNode* nd,
        aiVector3D* min,
        aiVector3D* max,
        aiMatrix4x4 &prev
        ){
        unsigned int n = 0, t;

        aiMatrix4x4 trafo = prev;
        trafo *= nd->mTransformation;

        for (; n < nd->mNumMeshes; ++n) {
            const aiMesh* mesh = sc->mMeshes[nd->mMeshes[n]];
            for (t = 0; t < mesh->mNumVertices; ++t) {

                aiVector3D tmp = mesh->mVertices[t];
                tmp *= trafo;

                min->x = std::min(min->x, tmp.x);
                min->y = std::min(min->y, tmp.y);
                min->z = std::min(min->z, tmp.z);

                max->x = std::max(max->x, tmp.x);
                max->y = std::max(max->y, tmp.y);
                max->z = std::max(max->z, tmp.z);
            }
        }

        for (n = 0; n < nd->mNumChildren; ++n) {
            get_bounding_box_for_node(sc, nd->mChildren[n], min, max, trafo);
        }
    }

    static void get_bounding_box(const aiScene *sc, aiVector3D* min, aiVector3D* max)
    {
        // set identity
        aiMatrix4x4 rootTransformation;

        min->x = min->y = min->z = 1e10f;
        max->x = max->y = max->z = -1e10f;
        get_bounding_box_for_node(sc, sc->mRootNode, min, max, rootTransformation);
    }
};

#endif // EXTERNALIMPORTER_H
