#include "trackball.hh"
#include <iostream>
#include <glm/gtx/string_cast.hpp>

#define R 1.f

static void computeNormVec3(const glm::vec2 &in, glm::vec3 &out) {
    float z;
    if (in.x * in.x + in.y * in.y <= R * R / 2) {
        z = sqrt(R * R - in.x * in.x - in.y * in.y);
    }
    else {
        z = R * R / 2.f / sqrt(in.x * in.x + in.y * in.y);
    }
    out.x = in.x;
    out.y = in.y;
    out.z = z;
    out = glm::normalize(out);
}

void computeRotation(const glm::vec2 &start, const glm::vec2 &end, glm::vec3 &N, float &angle) {
    // Reference: http://www.lubanren.net/weblog/post/283.html
    glm::vec3 a, b;
    computeNormVec3(start, a);
    computeNormVec3(end, b);

    N = glm::cross(a, b);

    float x = glm::angle(a, b) / glm::pi<float>();
    x = (-x * x + 2 * x);
    angle = x * glm::pi<float>();
}
