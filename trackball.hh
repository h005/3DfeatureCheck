#pragma once
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/string_cast.hpp>
#include <glm/gtx/vector_angle.hpp>

void computeRotation(const glm::vec2 &start, const glm::vec2 &end, glm::vec3 &N, float &angle);
