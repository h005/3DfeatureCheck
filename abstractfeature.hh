#ifndef ABSTRACTFEATURE_HH
#define ABSTRACTFEATURE_HH

#include "common.hh"

template <typename MeshT>
class AbstractFeature {
public:
    virtual void assignVertexValue() { assert(0);}
    virtual double compute(const glm::mat3 &K, const glm::mat3 &R, const glm::vec3 &t)
    {
        assert(0);
        return 1;
    }
};

#endif // ABSTRACTFEATURE_HH

