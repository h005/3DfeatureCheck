﻿#ifndef MEANCURVATURE_HH
#define MEANCURVATURE_HH

#include "common.hh"
#include "curvature.hh"
#include "colormap.hh"
#include "abstractfeature.hh"
#include <QDebug>

template <typename MeshT>
class MeanCurvature: public AbstractFeature<MeshT>
{
public:
    MeanCurvature(MeshT &in_mesh)
        : m_mesh(in_mesh), m_PropertyKeyword("Mean Curvature")
    {

        if(!m_mesh.get_property_handle(m_vPropHandle, m_PropertyKeyword))
            m_mesh.add_property(m_vPropHandle, m_PropertyKeyword);
        if(!m_mesh.get_property_handle(vertexBoundingArea, "area"))
            m_mesh.add_property(vertexBoundingArea, "area");

            OpenMesh::VPropHandleT<double> valuePerArea;

            m_mesh.add_property(valuePerArea);

            typename MeshT::VertexIter v_it, v_end(m_mesh.vertices_end());
            int time = 0;
            for (v_it = m_mesh.vertices_begin(); v_it != v_end; v_it++) {
                OpenMesh::VectorT<float,3> n;
                double area;
                if(time == 9)
                {
                    printf("for debug\n");
                }
                curvature::discrete_mean_curv_op<MeshT>(m_mesh, *v_it, n, area);

//                printf("meanCurvature....ok...%d\n",time++);

                // 每个顶点的平均曲率非负，因为其值为向量的长度
                m_mesh.property(valuePerArea, *v_it) = n.norm() / 2.0;
                Q_ASSERT(!std::isnan(m_mesh.property(valuePerArea, *v_it)));
                m_mesh.property(vertexBoundingArea, *v_it) = area;
                Q_ASSERT(!std::isnan(m_mesh.property(vertexBoundingArea, *v_it)));
                time++;
            }

            // TODO:
            // 优化缩放比的选择
            // 这里可能存在问题，由于包围面积过小，除法后结果值过大

            v_it = m_mesh.vertices_begin();
            double curvatureMax = m_mesh.property(valuePerArea, *v_it);
            v_it++;
            for (; v_it != v_end; v_it++)
                if (curvatureMax < m_mesh.property(valuePerArea, *v_it))
                    curvatureMax = m_mesh.property(valuePerArea, *v_it);

//            v_it = m_mesh.vertices_begin();
//            double maxNormal = abs(m_mesh.property(valuePerArea, *v_it));
//            if(curvatureMax)
//                maxNormal /= curvatureMax;
//            double minNormal = abs(m_mesh.property(valuePerArea, *v_it));
//            if(curvatureMax)
//                minNormal /= curvatureMax;

            // 如果一个mesh只有一个三角面，那么这三个顶点就都是边界点，从而每个顶点上的平均曲率都为0
            // 所以除之前看看curvatureMax是否为0
            for (v_it = m_mesh.vertices_begin(); v_it != v_end; v_it++) {
//                std::cout << "meanCurvature .. hi "<<std::endl;
                Q_ASSERT(!std::isnan(m_mesh.property(m_vPropHandle, *v_it)));
                if (curvatureMax)
                {
                    double tmp = abs(m_mesh.property(valuePerArea, *v_it)) / curvatureMax;
                    tmp = min(tmp,1.0);
                    m_mesh.property(m_vPropHandle, *v_it) = max(tmp,0.0);
//                    std::cout<<"meanCurvature ... "<< m_mesh.property(m_vPropHandle, *v_it) << std::endl;
//                    maxNormal = maxNormal > tmp ? maxNormal : tmp;
//                    minNormal = minNormal < tmp ? minNormal : tmp;
                }
//                else
//                    m_mesh.property(m_vPropHandle, *v_it) = m_mesh.property(valuePerArea, *v_it);
                Q_ASSERT(!std::isnan(m_mesh.property(m_vPropHandle, *v_it)));
            }

//            if(maxNormal - minNormal)
//            for(v_it = m_mesh.vertices_begin(); v_it != v_end; v_it++)
//                m_mesh.property(m_vPropHandle, *v_it) = (m_mesh.property(m_vPropHandle, *v_it) - minNormal) / (maxNormal - minNormal);

            m_mesh.remove_property(valuePerArea);
    }

    ~MeanCurvature()
    {
    }

    /**
     * @brief 将各个顶点的曲率结果，以颜色形式保存到绑定的mesh对象中，便于后续输出到文件
     */
    void assignVertexColor()
    {
        if (!m_mesh.has_vertex_colors())
            m_mesh.request_vertex_colors();

        typename MeshT::VertexIter v_it, v_end(m_mesh.vertices_end());
        for (v_it = m_mesh.vertices_begin(); v_it != v_end; v_it++) {
            double rgb[3];
            double v = m_mesh.property(m_vPropHandle, *v_it);
            ColorMap::jet(rgb, v, 0, 1);
            typename MeshT::Color color(rgb[0], rgb[1], rgb[2]);
            m_mesh.set_color(*v_it, color);
        }
    }

    double getMeanCurvature(std::vector<bool> isVertexVisible)
    {
        double res = 0.0;
        int index = 0;
        typename MeshT::VertexIter v_it, v_end(m_mesh.vertices_end());
        for (v_it = m_mesh.vertices_begin(); v_it != v_end; v_it++,index++)
            if(isVertexVisible[index]) {
                std::cout << "get meanCurvature debug index "<< index << std::endl;
                std::cout << m_mesh.property(m_vPropHandle, *v_it) << " " << m_mesh.property(vertexBoundingArea, *v_it) << std::endl;
                Q_ASSERT(!std::isnan(m_mesh.property(m_vPropHandle, *v_it)));
                Q_ASSERT(!std::isnan(m_mesh.property(vertexBoundingArea, *v_it)));
                Q_ASSERT(!std::isnan(m_mesh.property(m_vPropHandle, *v_it) * m_mesh.property(vertexBoundingArea, *v_it)));
                res += m_mesh.property(m_vPropHandle, *v_it) * m_mesh.property(vertexBoundingArea, *v_it);

//                qDebug()<<"mean curvature ... m_vPropHandle "<<m_mesh.property(m_vPropHandle, *v_it)<<endl;
//                qDebug()<<"mean curvature ... vertexBoundingArea "<<m_mesh.property(vertexBoundingArea, *v_it)<<endl;
            }

//        qDebug()<<"mean curvature ... res "<<res<<endl;
        return res;
    }

    void setMeanCurvature(std::vector<double> &meanCurvature)
    {
        typename MeshT::VertexIter v_it, v_end(m_mesh.vertices_end());
        for (v_it = m_mesh.vertices_begin(); v_it != v_end; v_it++)
            meanCurvature.push_back(m_mesh.property(m_vPropHandle, *v_it));
    }

    void setMeanCurvature(double* meanCurvature,std::vector<int> &verVec)
    {
        int i = 0;
        typename MeshT::VertexIter v_it, v_end(m_mesh.vertices_end());
        for (v_it = m_mesh.vertices_begin(); v_it != v_end; v_it++)
            meanCurvature[verVec[i++]] = m_mesh.property(m_vPropHandle, *v_it);
    }

    double compute(const glm::mat3 &K, const glm::mat3 &R, const glm::vec3 &t)
    {
        return 1;
    }

    double min(double a,double b)
    {
        return a < b ? a : b;
    }

    double max(double a,double b)
    {
        return a > b ? a : b;
    }

private:
    MeshT &m_mesh;
    OpenMesh::VPropHandleT<double> m_vPropHandle;
    OpenMesh::VPropHandleT<double> vertexBoundingArea;
    const char *m_PropertyKeyword;
};

#endif // MEANCURVATURE_HH

