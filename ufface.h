#ifndef UFFACE_H
#define UFFACE_H

#include <iostream>
#include <vector>
#include <set>

class UFface
{
public:
    UFface();
    UFface(std::vector<int> &indices);
    ~UFface();

    int* unionFinal(std::vector<int> &indices,std::vector<int> &cs);
    void free();


private:

    void shrink(std::vector<int> &indices);
    bool isSameFace(std::vector<int> &indices,int i1,int i2);

    int find(int p);
    void unionFace(int p,int q);

    int findCommonEdge(int p);
    void unionFaceCommonEdge(int p,int q);

    void setCateSet();
    void setCateCommonEdgeSet();

    void reArrange();

    char getRealtion(int i,int j);
    void setRelation(int i,int j,char val);

    void setRelation();
    void checkIn(int i,int j);

private:
    std::set<int> cateSet;
    std::set<int> cateSetCommonEdge;
    std::vector<int> *cate;

    int *sz;
    int *szCE;
    int **arrayFace;
    int *id;
    int *idCE;
    int NUM_FACE;
    char *relationGraph;
};

#endif // UFFACE_H
