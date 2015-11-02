#ifndef FACEGLHELPER
#define FACEGLHELPER

#include <assert.h>
#include <map>
#include <GL/glew.h>
#include <iostream>
#include <vector>
#include <QtDebug>

template<typename T>
class FaceGLHelper{
public:
    FaceGLHelper(const std::vector<GLfloat> &vertices,
                 const std::vector<GLfloat> &vertexNormals,
                 const std::vector<GLuint> &faceIndices)
    {
        m_vertices = vertices;
        m_vertexNormals = vertexNormals;
        m_facesIndices = faceIndices;
    }

    void init(GLuint vertexPositionID, GLuint vertexNormalID)
    {
        // 创建并绑定环境
        glGenVertexArrays(1, &m_vao);
        glBindVertexArray(m_vao);

        // vertex
        glGenBuffers(1, &m_vboVertex);
        glBindBuffer(GL_ARRAY_BUFFER, m_vboVertex);
        glBufferData(GL_ARRAY_BUFFER,
                     m_vertices.size() * sizeof(GLfloat),
                     &m_vertices[0],
                     GL_STATIC_DRAW);

        glVertexAttribPointer(vertexPositionID,
                              3,
                              GL_FLOAT,
                              GL_FALSE,
                              0,
                              NULL);
        glEnableVertexAttribArray (vertexPositionID);

        // normals
        glGenBuffers(1, &m_vboVertexNormal);
        glBindBuffer(GL_ARRAY_BUFFER, m_vboVertexNormal);
        glBufferData(GL_ARRAY_BUFFER,
                     m_vertexNormals.size() * sizeof(GLfloat),
                     &m_vertexNormals[0],
                     GL_STATIC_DRAW);

        glVertexAttribPointer(vertexNormalID,
                              3,
                              GL_FLOAT,
                              GL_FALSE,
                              0,
                              NULL);
        glEnableVertexAttribArray (vertexNormalID);


        // index
        glGenBuffers(1, &m_vboIndex);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,
                     m_vboIndex);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                     m_facesIndices.size() * sizeof(GLuint),
                     &m_facesIndices[0],
                     GL_STATIC_DRAW);

        numsToDraw = m_facesIndices.size();
        m_isInited = true;

    }

    void draw()
    {
        if (!m_isInited) {
            std::cout << "please call init() before draw()" << std::endl;
            assert(0);
        }

        // draw sphere
        glBindVertexArray(m_vao);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_vboIndex);
        glDrawElements(GL_TRIANGLES, numsToDraw, GL_UNSIGNED_INT, NULL);
    }

    void cleanup()
    {
        if (!m_isInited) {
            return;
        }
        if(m_vboVertex) {
            glDeleteBuffers(1, &m_vboVertex);
        }
        if(m_vboIndex) {
            glDeleteBuffers(1, &m_vboIndex);
        }
        if (m_vboVertexNormal)
            glDeleteBuffers(1, &m_vboVertexNormal);
        if (m_vao) {
            glDeleteVertexArrays(1, &m_vao);
        }

        m_isInited = false;
        m_vao = 0;
        m_vboVertex = 0;
        m_vboIndex = 0;
    }


private:
    std::vector<GLfloat> m_vertices;
    std::vector<GLfloat> m_vertexNormals;
    std::vector<GLuint> m_facesIndices;
    bool m_isInited;
    GLuint m_vao = 0, m_vboVertex = 0, m_vboIndex = 0,m_vboVertexNormal = 0;
    int numsToDraw;
};

#endif // MESHGLHELPER

