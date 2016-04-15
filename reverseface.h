#ifndef REVERSEFACE_H
#define REVERSEFACE_H

#include <iostream>
#include <vector>
#include <map>
#include <algorithm>
#include <string.h>
#include <set>
#include <QtGlobal>
#include <QString>

class ReverseFace
{
public:
    ReverseFace();
    ReverseFace(std::vector<int> &indices);
    ~ReverseFace();

    int* reverseFace(std::vector<int> &indices,std::vector<int> &cs, std::vector< std::set<int> > &cate);

private:

    void shrink(std::vector<int> &indices);

    bool isSameFace(std::vector<int> &indices, int i1,int i2);

    void setRelation(std::vector<int> &indices);

    void reverse(std::vector<int> &indices);

    bool dfsReverse(std::vector<int> &indices, char *visit, int ind, int count, std::set<int> &tmpSet);

    // check weather two faces has the same order
    // if has the same order then return false
    bool checkOrder(std::vector<int> &indices,int ind1,int ind2);

    int dfs(char *visit, int ind, int count);

private:

    int NUM_FACE;

    std::map<std::pair<int,int>,int> adjTable;

    // true adj table
    std::vector< std::vector<int> > adjt;

    int *sz;

    int *id;

};

#endif // REVERSEFACE_H
