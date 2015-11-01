#ifndef EXTERNALIMPORTER_HH
#define EXTERNALIMPORTER_HH

#include "common.hh"
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <assimp/DefaultLogger.hpp>
#include <assimp/LogStream.hpp>
#include "ufface.h"
#include <QString>
#include "predefine.h"

template <typename MeshT>
class ExternalImporter
{
public:
    /**
     * @brief 读取一个外部文件，类似于OpenMesh::IO::read_mesh
     * @param mesh 结果存放位置
     * @param path 文件路径
     * @return 是否读取成功
     * @see OpenMesh::IO::read_mesh
     */
    bool read_mesh(MeshT &mesh,const char *path)
    {
        Assimp::Importer importer;
        const aiScene *scene = importer.ReadFile(path,aiProcessPreset_TargetRealtime_Quality);

        if(!scene)
            return false;

        int count = 0;

        getPointFace_h005(scene,scene->mRootNode,glm::mat4(),vertices,indices,count);

#ifndef CHECK

        UFface *ufface = new UFface(indices);

        id = ufface->unionFinal(indices,cateSet);
        ufface->free();
        std::cout << "union ... done "<<std::endl;
#endif
        buildMesh_h005(vertices,indices,mesh);

        std::cout<<"Assimp Importer: "<<count<<" Meshes Loaded."<<std::endl;
        return true;
    }

    int outputMesh(MeshT &mesh,QString fileName)
    {
        fileName.append(QString(".off"));
        try
        {
          if ( !OpenMesh::IO::write_mesh(mesh, fileName.toStdString().c_str()) )
          {
            std::cerr << "Cannot write mesh to file " << fileName.toStdString()<< std::endl;
            return 1;
          }
        }
        catch( std::exception& x )
        {
          std::cerr << x.what() << std::endl;
          return 1;
        }
    }

    void setMeshVector(std::vector<MeshT> &mesh,std::vector<std::vector<int>> &indicesArray)
    {
        // 设置不同mesh的面的索引
        setIndiceMesh(indices.size()/3);

        // indiceMesh 存储各个mesh中face的indices
        // indicesArray 存储各个mesh中vertex的indices
        // 其中一个face对应于三个vertex，仅仅是个扩充而已
        indicesArray.clear();

        printf("setMeshVector...indiceMesh size %d\n",indiceMesh.size());

        for(int i=0;i<indiceMesh.size();i++)
        {
            std::vector<int> tmpArray;
            for(int j=0;j<indiceMesh[i].size();j++)
            {
                tmpArray.push_back(indices[indiceMesh[i][j]*3]);
                tmpArray.push_back(indices[indiceMesh[i][j]*3+1]);
                tmpArray.push_back(indices[indiceMesh[i][j]*3+2]);
            }
            indicesArray.push_back(tmpArray);
        }


        printf("setMeshVector....%d\n",indices.size()/3);
        printf("setMeshVector...cateSet size....%d\n",cateSet.size());
        printf("setMeshVector...verticeSize...%d\n",vertices.size());
        printf("setMeshVector...indces Size...%d\n",indices.size());

        /*
         * 对每个mesh将vertex，push进去，然后将face放进去
         * indiceSet 存储着各个vertex的索引，存储着哪些vertex出现过，需要push进mesh中
         * tmpMesh 当前构建的mesh
         * vHandle push进tmpMesh的vHandle
         * tmpIndex[*it] 值为*it的vertex索引在vHandle出现在第几次
        */
        for(int i=0;i<cateSet.size();i++)
        {
            MeshT tmpMesh;
            std::vector<typename MeshT::VertexHandle> vHandle;
            std::vector<MyMesh::VertexHandle> face_vhandles;
//            printf("setMeshVector...indiceMesh[%d] size %d\n",i,indiceMesh[i].size());
            std::set<int> indiceSet;
            for(int j=0;j<indiceMesh[i].size();j++)
            {
                indiceSet.insert(indices[indiceMesh[i][j]*3]);
                indiceSet.insert(indices[indiceMesh[i][j]*3+1]);
                indiceSet.insert(indices[indiceMesh[i][j]*3+2]);
            }

            std::set<int>::iterator it = indiceSet.begin();

            for(;it!=indiceSet.end();it++)
                vHandle.push_back(tmpMesh.add_vertex(vertices[*it]));
            it--;
            int max = *it;
            int *tmpIndex = new int[max];
            memset(tmpIndex,-1,sizeof(int)*max);
            it = indiceSet.begin();
            int tmpCount = 0;
            for(;it!=indiceSet.end();it++)
                tmpIndex[*it] = tmpCount++;

            for(int j=0;j<indiceMesh[i].size();j++)
            {
                face_vhandles.clear();
                it = indiceSet.find(indices[indiceMesh[i][j]*3]);
                face_vhandles.push_back(vHandle[tmpIndex[*it]]);
                it = indiceSet.find(indices[indiceMesh[i][j]*3+1]);
                face_vhandles.push_back(vHandle[tmpIndex[*it]]);
                it = indiceSet.find(indices[indiceMesh[i][j]*3+2]);
                face_vhandles.push_back(vHandle[tmpIndex[*it]]);
                tmpMesh.add_face(face_vhandles);
            }

            mesh.push_back(tmpMesh);
        }

        printf("setMeshVector....done\n");

    }

private:
    static void recursive_create(const aiScene *sc,
                                 const aiNode *nd,
                                 const glm::mat4 &inheritedTransformation,
                                 MeshT &openMesh,
                                 int &count)
    {
        assert(nd && sc);
        unsigned int n = 0;

        glm::mat4 mTransformation = glm::transpose(glm::make_mat4((float *)&nd->mTransformation));
        glm::mat4 absoluteTransformation = inheritedTransformation * mTransformation;

        count += nd->mNumMeshes;

        for (; n < nd->mNumMeshes; ++n)
        {
            // 一个aiNode中存有其mesh的索引，
            // 在aiScene中可以用这个索引拿到真正的aiMesh
            const struct aiMesh* mesh = sc->mMeshes[nd->mMeshes[n]];

            // 将所有点变换后，加入OpenMesh结构中，并保存它们的索引
            std::vector<typename MeshT::VertexHandle> vHandle;
//            HasPosition() position 和 vertex 是什么区别

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
                        openMesh.add_face(fHandle);
                    }
                    else
                        std::cout<<"mesh face..."<< i <<std::endl;
                }
            }
        }


        // create all children

        for (n = 0; n < nd->mNumChildren; ++n)
            recursive_create(sc, nd->mChildren[n], absoluteTransformation, openMesh, count);

    }

    static void buildMesh_h005(std::vector<typename MeshT::Point> &vertices,
                        std::vector<int> &indices,
                        MeshT &mesh)
    {
        printf("buildMesh_h005 vertices size %d\n",vertices.size());
        printf("buildMesh_h005 indices size %d\n",indices.size());
        std::vector<typename MeshT::VertexHandle> vHandle;
        for(int i=0;i<vertices.size();i++)
            vHandle.push_back(mesh.add_vertex(vertices[i]));
        std::vector<MyMesh::VertexHandle> face_vhandles;
        std::cout<<"buildMesh_h005...indices size "<<indices.size()<<std::endl;
        for(int i=0;i<indices.size();i+=3)
        {
//            printf("buildMesh_h005 face %d %d %d\n",indices[i],indices[i+1],indices[i+2]);
            face_vhandles.clear();
            face_vhandles.push_back(vHandle[indices[i]]);
            face_vhandles.push_back(vHandle[indices[i+1]]);
            face_vhandles.push_back(vHandle[indices[i+2]]);
//            printf("buildMesh_h005 ... %d %d %d %d\n",i,indices[i],indices[i+1],indices[i+2]);
            mesh.add_face(face_vhandles);
        }
    }

    static void getPointFace_h005(const aiScene *sc,
                     const aiNode *nd,
                     const glm::mat4 &inheritedTransformation,
                     std::vector<typename MeshT::Point> &vertices,
                     std::vector<int> &indices,
                     int &count)
    {
        assert(nd && sc);
        unsigned int n = 0;

        glm::mat4 mTransformation = glm::transpose(glm::make_mat4((float *)&nd->mTransformation));
        glm::mat4 absoluteTransformation = inheritedTransformation * mTransformation;

        count += nd->mNumMeshes;

        for (; n < nd->mNumMeshes; ++n)
        {
            // 一个aiNode中存有其mesh的索引，
            // 在aiScene中可以用这个索引拿到真正的aiMesh
            const struct aiMesh* mesh = sc->mMeshes[nd->mMeshes[n]];

            // 将所有点变换后，加入OpenMesh结构中，并保存它们的索引
//            std::vector<typename MeshT::VertexHandle> vHandle;
            std::vector<typename MeshT::Point> vectorPoint;
//            HasPosition() position 和 vertex 是什么区别

            if(mesh->HasPositions()) {
                for(uint32_t i = 0; i < mesh->mNumVertices; ++i) {
                    glm::vec3 position(mesh->mVertices[i].x, mesh->mVertices[i].y, mesh->mVertices[i].z);
                    glm::vec4 absolutePosition = absoluteTransformation * glm::vec4(position, 1.f);

                    typename MeshT::Point point(absolutePosition.x, absolutePosition.y, absolutePosition.z);
                    vectorPoint.push_back(point);
//                    vHandle.push_back(openMesh.add_vertex(point));
                }
            }

            if(mesh->HasFaces() && vectorPoint.size() > 0) {

                int base = vertices.size();
                for(int i=0;i<vectorPoint.size();i++)
                    vertices.push_back(vectorPoint[i]);

                // 一个face代表一个面（暂时只考虑三角形，其余类型pass），其存储着各个顶点的索引
                // 可以根据索引到mesh->mVertices[]中找到对应顶点的数据(x, y, z)
                // 不能只考虑三角形，这样会导致面片的不连续，从而出错
                for(uint32_t i = 0; i < mesh->mNumFaces; ++i)
                {
                    for(uint32_t j = 0; j < mesh->mFaces[i].mNumIndices - 2; j++)
                    {
                        indices.push_back(base + mesh->mFaces[i].mIndices[0]);
                        indices.push_back(base + mesh->mFaces[i].mIndices[j+1]);
                        indices.push_back(base + mesh->mFaces[i].mIndices[j+2]);
                    }
                }

            }
        }


        // create all children

        for (n = 0; n < nd->mNumChildren; ++n)
            getPointFace_h005(sc, nd->mChildren[n], absoluteTransformation, vertices, indices, count);
    }

    void setIndiceMesh(int length)
    {
        // release
        for(int i=0;i<indiceMesh.size();i++)
            for(int j=0;j<indiceMesh[i].size();j++)
                std::vector<int>().swap(indiceMesh[i]);
        for(int i=0;i<indiceMesh.size();i++)
            std::vector<std::vector<int>>().swap(indiceMesh);

        indiceMesh.clear();
        // initial indiceMesh
        int len = cateSet.size();
        for(int i=0;i<len;i++)
            indiceMesh.push_back(std::vector<int>());

        for(int i=0;i<len;i++)
        {
            for(int j=0;j<length;j++)
            {
                if(find(id,j) ==  cateSet[i])
                    indiceMesh[i].push_back(j);
            }
        }

    }

    int find(int* id,int p)
    {
        while(p != id[p])
        {
            id[p] = id[id[p]];
            p = id[p];
        }
        return p;
    }



    // 可能会有很多个mesh，存储每个mesh中face的indices
    std::vector<std::vector<int>> indiceMesh;
    // 并查集合并之后的结果
    int *id;
    // 并查集合并之后的集合索引
    std::vector<int> cateSet;
    // 模型文件中的点
    std::vector<typename MeshT::Point> vertices;
    // 面的索引
    std::vector<int> indices;

};



#endif // EXTERNALIMPORTER_HH

