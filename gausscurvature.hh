#ifndef GAUSSCURVATURE_H
#define GAUSSCURVATURE_H

#include <limits>

#include "common.hh"
#include "Curvature.hh"
#include "colormap.hh"
#include "abstractfeature.hh"

template <typename MeshT>
class GaussCurvature: public AbstractFeature<MeshT>
{
public:
    GaussCurvature(MeshT &in_mesh)
        : m_mesh(in_mesh), m_PropertyKeyword("Gauss Curvature")
    {
        if(!m_mesh.get_property_handle(m_vPropHandle, m_PropertyKeyword))
            m_mesh.add_property(m_vPropHandle, m_PropertyKeyword);


        if(!m_mesh.get_property_handle(vertexBoundingArea, "area"))
        {
            std::cout<<"gauss .... get handle error "<<std::endl;
        }
        else
            std::cout<<"gauss .... get handle "<<std::endl;

        typename MeshT::VertexIter v_it, v_end(m_mesh.vertices_end());
        for (v_it = m_mesh.vertices_begin(); v_it != v_end; v_it++) {
            double curv = curvature::gauss_curvature<MeshT>(m_mesh, *v_it);
            m_mesh.property(m_vPropHandle, *v_it) = abs(curv);
        }

        v_it = m_mesh.vertices_begin();
        double curvatureMax = m_mesh.property(m_vPropHandle, *v_it);
        v_it ++;
        // 归一化
        for (; v_it != v_end; v_it++)
            if (curvatureMax < m_mesh.property(m_vPropHandle, *v_it))
                curvatureMax = m_mesh.property(m_vPropHandle, *v_it);

        std::cout<<"gauss .... curvatureMax "<<curvatureMax<<std::endl;

        v_it = m_mesh.vertices_begin();
        double maxNormal = abs(m_mesh.property(m_vPropHandle, *v_it));
        if(curvatureMax)
            maxNormal /= curvatureMax;
        double minNormal = abs(m_mesh.property(m_vPropHandle, *v_it));
        if(curvatureMax)
            minNormal /= curvatureMax;
        // 如果一个mesh只有一个三角面，那么这三个顶点就都是边界点，从而每个顶点上的平均曲率都为0
        // 所以除之前看看curvatureMax是否为0
        for (; v_it != v_end; v_it++) {
            Q_ASSERT(!std::isnan(m_mesh.property(m_vPropHandle, *v_it)));
            if (curvatureMax)
            {
                double tmp = abs(m_mesh.property(m_vPropHandle, *v_it)) / curvatureMax;
                m_mesh.property(m_vPropHandle, *v_it) = tmp;
                maxNormal = maxNormal > tmp ? maxNormal : tmp;
                minNormal = minNormal < tmp ? minNormal : tmp;
            }
            Q_ASSERT(!std::isnan(m_mesh.property(m_vPropHandle, *v_it)));
        }

        if(maxNormal - minNormal)
            for(v_it = m_mesh.vertices_begin(); v_it != v_end; v_it++)
                m_mesh.property(m_vPropHandle, *v_it) = (m_mesh.property(m_vPropHandle, *v_it) - minNormal) / (maxNormal - minNormal);

    }

    ~GaussCurvature()
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

        double curvatureMax = -1, curvatureMin = std::numeric_limits<double>::max();
        for (v_it = m_mesh.vertices_begin(); v_it != v_end; v_it++) {
            if (curvatureMax < m_mesh.property(m_vPropHandle, *v_it)) {
                curvatureMax = m_mesh.property(m_vPropHandle, *v_it);
            }
            if (curvatureMin > m_mesh.property(m_vPropHandle, *v_it)) {
                curvatureMin = m_mesh.property(m_vPropHandle, *v_it);
            }
        }

        for (v_it = m_mesh.vertices_begin(); v_it != v_end; v_it++) {
            double rgb[3];
            double v = m_mesh.property(m_vPropHandle, *v_it);
            ColorMap::jet(rgb, v, curvatureMin, curvatureMax);
            typename MeshT::Color color(rgb[0], rgb[1], rgb[2]);
            m_mesh.set_color(*v_it, color);
        }
    }

    double getGaussianCurvature(std::vector<bool> isVertexVisible)
    {
        double res = 0.0;
        int index = 0;
        typename MeshT::VertexIter v_it, v_end(m_mesh.vertices_end());
        for (v_it = m_mesh.vertices_begin(); v_it != v_end; v_it++,index++)
            if(isVertexVisible[index])
                res += m_mesh.property(m_vPropHandle, *v_it) * m_mesh.property(vertexBoundingArea, *v_it);
        return res;
    }

    double compute(const glm::mat3 &K, const glm::mat3 &R, const glm::vec3 &t)
    {
        return 1;
    }

private:
    MeshT &m_mesh;
    OpenMesh::VPropHandleT<double> m_vPropHandle;
    const char *m_PropertyKeyword;
    OpenMesh::VPropHandleT<double> vertexBoundingArea;
};

#endif // GAUSSCURVATURE_H
