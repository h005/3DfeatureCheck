﻿#ifndef EXTERNALIMPORTER_HH
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
#include <iostream>
#include <fstream>
#include "reverseface.h"

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

//        QString debugPath = "/home/h005/Documents/kxm/debug/beforeShrink.txt";
//        std::ofstream out;
//        out.open(debugPath.toStdString().c_str());
//        if(!out.is_open())
//        {
//            std::cout << "before open error"<< std::endl;
//        }
//        out << indices.size() << std::endl;
//        for(int i=0;i<indices.size();i++)
//        {
//            out << indices[i] << std::endl;
//        }
//        out.close();
//        std::cout << "before shrink done" << std::endl;
//        std::cout << "indices size "<< indices.size() << std::endl;


//        QString uffacePath = "/home/h005/Documents/kxm/debug/uffaceData.txt";
//        indices.clear();
//        std::ifstream in(uffacePath.toStdString().c_str());
//        int tmpind;
//        while(in >> tmpind)
//            indices.push_back(tmpind);
//        in.close();


//            UFface *ufface = new UFface(indices);

//            id = ufface->unionFinal(indices,cateSet);
//            ufface->free();

        ReverseFace *reverse = new ReverseFace(indices);
        id = reverse->reverseFace(indices,cateSet,cate);

        qDebug() << "indices size " << indices.size() << endl;
#endif
        buildMesh_h005(vertices,indices,mesh);
        qDebug() << "build mesh ok ........................................................................................." << endl;
        // output the file
//        outputMesh(mesh,"E:/ViewPoint/kxm/201511231826/kxm");

//        outputMesh(mesh,"D:/viewpoint/bigBen/201511271435/bigBen");

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

    void setMeshVector(std::vector<MeshT> &mesh,std::vector< std::vector<int> > &indicesArray)
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

        qDebug()<<"setMeshVector ... "<<cateSet.size()<<endl;
        for(int i=0;i<cateSet.size();i++)
        {
            if(i == 2)
                std::cout << "debug" << std::endl;
            std::cout << "cateSet "<<i<<"/"<< cateSet.size() << std::endl;
//            std::cout << "debug .... ok 0"<< std::endl;
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
//            std::cout << "debug .... ok 1"<< std::endl;
            std::set<int>::iterator it = indiceSet.begin();

            for(;it!=indiceSet.end();it++)
                vHandle.push_back(tmpMesh.add_vertex(vertices[*it]));
            it--;
            int max = *it;
//            std::cout << "debug max "<<max<<std::endl;
//            std::cout << "debug indiceSet size "<<indiceSet.size()<< std::endl;
            int *tmpIndex = new int[max+1];
            memset(tmpIndex,-1,sizeof(int)*max+1);
            it = indiceSet.begin();
            int tmpCount = 0;
            for(;it!=indiceSet.end();it++) {
//                std::cout << "debug ... indiceSet " << *it << std::endl;
                tmpIndex[*it] = tmpCount++;
               }
//            std::cout << "debug .... ok 2"<< std::endl;
            for(int j=0;j<indiceMesh[i].size();j++)
            {
//                std::cout << "index " << j << std::endl;
//                if(j == 5)
//                    std::cout << "debug" << std::endl;
                face_vhandles.clear();
                it = indiceSet.find(indices[indiceMesh[i][j]*3]);
                face_vhandles.push_back(vHandle[tmpIndex[*it]]);
                it = indiceSet.find(indices[indiceMesh[i][j]*3+1]);
                face_vhandles.push_back(vHandle[tmpIndex[*it]]);
                it = indiceSet.find(indices[indiceMesh[i][j]*3+2]);
                face_vhandles.push_back(vHandle[tmpIndex[*it]]);
                tmpMesh.add_face(face_vhandles);
            }
//            std::cout << "debug .... ok 3"<< std::endl;
//            std::cout << "debug .... size "<<mesh.size() << std::endl;

            std::cout << "debug...  mesh num vertices " << tmpMesh.n_vertices () << std::endl;
            mesh.push_back(tmpMesh);

//            std::cout << "debug .... ok 4"<< std::endl;
            delete tmpIndex;
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
                        typename MeshT::FaceHandle fh = openMesh.add_face(fHandle);
                        if(!fh.isvalid())
                        {
                            fHandle[2] = vHandle[mesh->mFaces[i].mIndices[0]];
                            fHandle[1] = vHandle[mesh->mFaces[i].mIndices[1]];
                            fHandle[0] = vHandle[mesh->mFaces[i].mIndices[2]];
                            if(!openMesh.add_face(fHandle).is_valid())
                                std::cout << "drop" << std::endl;
                            else
                                std::cout << "reversed" << std::endl;
                        }
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
            printf("buildMesh_h005 face %d %d %d %d\n",i,indices[i],indices[i+1],indices[i+2]);
            face_vhandles.clear();
            face_vhandles.push_back(vHandle[indices[i]]);
            face_vhandles.push_back(vHandle[indices[i+1]]);
            face_vhandles.push_back(vHandle[indices[i+2]]);
//            printf("buildMesh_h005 ... %d %d %d %d\n",i,indices[i],indices[i+1],indices[i+2]);
            mesh.add_face(face_vhandles);
//            typename MeshT::FaceHandle fh = mesh.add_face(face_vhandles);
//            if(!fh.is_valid())
//            {
//                face_vhandles[2] = vHandle[indices[i]];
//                face_vhandles[1] = vHandle[indices[i+1]];
//                face_vhandles[0] = vHandle[indices[i+2]];
//                if(!mesh.add_face(face_vhandles).is_valid())
//                    std::cout << "drop" << std::endl;
//                else
//                    std::cout << "reversed" << std::endl;
//            }
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
                for(int i = 0; i < (int)mesh->mNumFaces; ++i) // 要改成int，并强制类型转换，不然会越界
                {
//                    std::cout << "mesh  faces num "<< mesh->mFaces[i].mNumIndices << std::endl;
                    for(int j = 0; j < (int)mesh->mFaces[i].mNumIndices - 2; j++) // 要改成int，并强制类型转换，不然会越界
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
            std::vector< std::vector<int> >().swap(indiceMesh);

        indiceMesh.clear();
        // initial indiceMesh
        int len = cateSet.size();
        for(int i=0;i<len;i++)
            indiceMesh.push_back(std::vector<int>());

        for(int i=0;i<len;i++)
        {
            std::set<int>::iterator it = cate[i].begin();
            for(;it!=cate[i].end();it++)
                indiceMesh[i].push_back(*it);
        }


        // ufface code
//        for(int i=0;i<len;i++)
//        {
//            for(int j=0;j<length;j++)
//            {
//                if(find(id,j) ==  cateSet[i])
//                    indiceMesh[i].push_back(j);
//            }
//        }

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
    std::vector< std::vector<int> > indiceMesh;
    // 并查集合并之后的结果
    int *id;
    // 并查集合并之后的集合索引
    // 面片绕序整理之后的类别
    std::vector<int> cateSet;
    // 面片绕序整理之后每个类别所对应的面片ID
    std::vector< std::set<int> > cate;
    // 模型文件中的点
    std::vector<typename MeshT::Point> vertices;
    // 面的索引
    std::vector<int> indices;

};



#endif // EXTERNALIMPORTER_HH

