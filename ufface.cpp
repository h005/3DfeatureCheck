#include "ufface.h"

UFface::UFface()
{

}

UFface::UFface(std::vector<int> &indices)
{

    shrink(indices);
    std::cout<<"shrink done..."<<std::endl;
    NUM_FACE = indices.size()/3;
    arrayFace = new int*[NUM_FACE];
    for(int i=0;i<NUM_FACE;i++)
    {
        arrayFace[i] = new int[3];
        arrayFace[i][0] = indices[i*3];
        arrayFace[i][1] = indices[i*3+1];
        arrayFace[i][2] = indices[i*3+2];
    }
    sz = new int[NUM_FACE];
    memset(sz,0,sizeof(int)*NUM_FACE);
    szCE = new int[NUM_FACE];
    memset(szCE,0,sizeof(int)*NUM_FACE);
    id = new int[NUM_FACE];
    idCE = new int[NUM_FACE];
    for(int i=0;i<NUM_FACE;i++)
        id[i] = idCE[i] = i;
    relationGraph = new char[NUM_FACE*(NUM_FACE-1)/2];

    memset(relationGraph,0,sizeof(char)*NUM_FACE*(NUM_FACE-1)/2);

    printf("UFface initial... done\n");

//    for(int i=0;i<NUM_FACE;i++)
//    {
//        relationGraph[i] = new char[NUM_FACE];
//        memset(relationGraph[i],0,sizeof(char)*NUM_FACE);
//    }

}

UFface::~UFface()
{
    cateSet.clear();
    cateSetCommonEdge.clear();
    delete cate;
    delete sz;
    delete szCE;
    for(int i=0;i<NUM_FACE;i++)
        delete arrayFace[i];
    delete relationGraph;
    delete idCE;

}

int UFface::find(int p)
{
    while(p != id[p])
    {
        id[p] = id[id[p]];
        p = id[p];
    }
    return p;
}

void UFface::unionFace(int p, int q)
{
    int fp = find(p);
    int fq = find(q);
    if(fp == fq)
        return;
    if(sz[fp] > sz[fq])
    {
        id[fq] = fp;
        sz[fp] += sz[fq];
    }
    else
    {
        id[fp] = fq;
        sz[fq] += sz[fp];
    }
}

int UFface::findCommonEdge(int p)
{
    while(p != idCE[p])
    {
        idCE[p] = idCE[idCE[p]];
        p = idCE[p];
    }
    return p;
}

void UFface::unionFaceCommonEdge(int p, int q)
{
    int fp = findCommonEdge(p);
    int fq = findCommonEdge(q);
    if(fp == fq)
        return;
    if(szCE[fp] > szCE[fq])
    {
        idCE[fq] = fp;
        szCE[fp] += szCE[fq];
    }
    else
    {
        idCE[fp] = fq;
        szCE[fq] += szCE[fp];
    }
}

int* UFface::unionFinal(std::vector<int> &indices,std::vector<int> &cs)
{
    setRelation();
//    freopen("realtion.txt","w",stdout);
//    for(int i=0;i<NUM_FACE*(NUM_FACE-1)/2;i++)
//        printf("%d %d\n",(int)relationGraph[i],i);
//    freopen("CON","w",stdout);
    printf("UFface setRelation done...\n");
    printf("unionFinal .... indices size %d\n",indices.size());
    for(int i=1;i<NUM_FACE;i++)
        for(int j=0;j<i;j++)
        {
            char tmpRelation = getRealtion(i,j);
            if(tmpRelation%2==1)
            {
                unionFace(i,j);
                unionFaceCommonEdge(i,j);
            }
            else if((tmpRelation>>1)%2==1)
                unionFaceCommonEdge(i,j);
        }
    setCateCommonEdgeSet();
    setCateSet();

    printf("unionFinal... num of Categories: %d\n",cateSet.size());
    printf("unionFinal... num of CategoriesCommonEdge: %d\n",cateSetCommonEdge.size());

    while(cateSet.size() != cateSetCommonEdge.size())
    {
        reArrange();
        setRelation();
        for(int i=1 ; i<NUM_FACE ; i++)
            for(int j=0 ; j<i ; j++)
                if(getRealtion(i,j)%2==1)
                    unionFace(i,j);
        setCateSet();
    }

    cs.clear();
    std::set<int>::iterator it = cateSet.begin();
    for(;it!=cateSet.end();it++)
        cs.push_back(*it);
    indices.clear();
    for(int i=0;i<3*NUM_FACE;i++)
        indices.push_back(arrayFace[i/3][i%3]);

    return id;
/*
    if(cateSet.size()==cateSetCommonEdge.size())
    {
        cs.clear();
        std::set<int>::iterator it = cateSet.begin();
        for(;it!=cateSet.end();it++)
            cs.push_back(*it);

        printf("unionFinal...cs size %d\n",cs.size());
        printf("unionFinal...id size %d\n",NUM_FACE);

        return id;
    }
    reArrange();
    do{
        setRelation();
        for(int i=1;i<NUM_FACE;i++)
            for(int j=0;j<i;j++)
                if(relationGraph[i][j]%2==1)
                    unionFace(i,j);
        setCateSet();
        printf("unionFinal... num of Categories: %d\n",cateSet.size());
        printf("unionFinal... num of CategoriesCommonEdge: %d\n",cateSetCommonEdge.size());
        if(cateSet.size() == cateSetCommonEdge.size())
        {
            cs.clear();
            std::set<int>::iterator it = cateSet.begin();
            for(;it!=cateSet.end();it++)
                cs.push_back(*it);
            indices.clear();
            for(int i=0;i<NUM_FACE*3;i++)
                indices.push_back(arrayFace[i/3][i%3]);
            return id;
        }
        else
            reArrange();
//        getchar();
    }while(cateSet.size() != cateSetCommonEdge.size());

    std::cout<<"union...."<<std::endl;
//    this->unionFinal();
    indices.clear();;
    for(int i=0;i<3*NUM_FACE;i++)
        indices.push_back(arrayFace[i/3][i%3]);

    cs.clear();
    setCateSet();
    std::set<int>::iterator it = cateSet.begin();
    for(;it!=cateSet.end();it++)
        cs.push_back(*it);

    return id;
*/
}

void UFface::free()
{
//    cateSet.clear();
//    cateSetCommonEdge.clear();
//    delete cate;
    delete sz;
    delete szCE;
    for(int i=0;i<NUM_FACE;i++)
        delete arrayFace[i];
    delete relationGraph;
    delete idCE;
}

bool UFface::isSameFace(std::vector<int> &indices, int i1, int i2)
{
    int face0[3];
    int face1[3];
    int min01 = indices[i1] < indices[i1+1] ? indices[i1] : indices[i1+1];
    int max01 = indices[i1] + indices[i1+1] - min01;
    face0[0] = indices[i1+2] < min01 ? indices[i1+2] : min01;
    face0[2] = indices[i1+2] > max01 ? indices[i1+2] : max01;
    face0[1] = indices[i1] + indices[i1+1] + indices[i1+2]
            - face0[0] - face0[2];
    min01 = indices[i2] < indices[i2+1] ? indices[i2] : indices[i2+1];
    max01 = indices[i2] + indices[i2+1] - min01;
    face1[0] = indices[i2+2] < min01 ? indices[i2+2] : min01;
    face1[2] = indices[i2+2] > max01 ? indices[i2+2] : max01;
    face1[1] = indices[i2] + indices[i2+1] + indices[i2+2]
            - face1[0] - face1[2];
    for(int i=0;i<3;i++)
        if(face0[i]!=face1[i])
            return false;
    return true;

}

void UFface::shrink(std::vector<int> &indices)
{
    for(int i=0;i<indices.size();i+=3)
        for(int j=i+3;j<indices.size();j+=3)
            if(isSameFace(indices,i,j))
            {
                indices.erase(indices.begin()+j,indices.begin()+j+3);
                j-=3;
            }

}

void UFface::setRelation()
{
    for(int i=1;i<NUM_FACE;i++)
        for(int j=0;j<i;j++)
            checkIn(i,j);
}

void UFface::setCateSet()
{
    cateSet.clear();
    for(int i=0;i<NUM_FACE;i++)
        cateSet.insert(find(i));
}

void UFface::setCateCommonEdgeSet()
{
    cateSetCommonEdge.clear();
    for(int i=0;i<NUM_FACE;i++)
        cateSetCommonEdge.insert(findCommonEdge(i));

}

void UFface::reArrange()
{
    cate = new std::vector<int>[cateSetCommonEdge.size()];
    std::set<int>::iterator it = cateSetCommonEdge.begin();
    int index = 0;
    for(;it!=cateSetCommonEdge.end();it++,index++)
        for(int i=0;i<NUM_FACE;i++)
            if(findCommonEdge(i)==*it)
                cate[index].push_back(i);

    for(int i=0;i < cateSetCommonEdge.size();i++)
    {
        int pid = find(cate[i][0]);
        for(int j=1; j < cate[i].size();j++)
        {
            if(find(cate[i][j]) != pid)
            {
                int tmp = arrayFace[cate[i][j]][0];
                arrayFace[cate[i][j]][0] = arrayFace[cate[i][j]][2];
                arrayFace[cate[i][j]][2] = tmp;
            }
        }
    }

}

char UFface::getRealtion(int i, int j)
{
    int pla = (i - 1) * i / 2 + j;
    return relationGraph[pla];
}

void UFface::setRelation(int i, int j, char val)
{
    int pla = (i - 1) * i / 2 + j;
    relationGraph[pla] = val;
}

void UFface::checkIn(int i, int j)
{
    unsigned long long num0[3]={
        (unsigned long long)arrayFace[i][0]<<32 | (unsigned long long)arrayFace[i][1],
        (unsigned long long)arrayFace[i][1]<<32 | (unsigned long long)arrayFace[i][2],
        (unsigned long long)arrayFace[i][2]<<32 | (unsigned long long)arrayFace[i][0]
    };
    unsigned long long num1[3]={
        (unsigned long long)arrayFace[j][1]<<32 | (unsigned long long)arrayFace[j][0],
        (unsigned long long)arrayFace[j][2]<<32 | (unsigned long long)arrayFace[j][1],
        (unsigned long long)arrayFace[j][0]<<32 | (unsigned long long)arrayFace[j][2]
    };
    unsigned long long num2[3]={
        (unsigned long long)arrayFace[j][0]<<32 | (unsigned long long)arrayFace[j][1],
        (unsigned long long)arrayFace[j][1]<<32 | (unsigned long long)arrayFace[j][2],
        (unsigned long long)arrayFace[j][2]<<32 | (unsigned long long)arrayFace[j][0]
    };
    for(int i0 = 0; i0 < 3; i0++)
        for(int j0 = 0; j0 < 3 ; j0++)
        {
            if(num0[i0]==num1[j0])
                setRelation(i,j,3);
            else if(num0[i0]==num2[j0])
                setRelation(i,j,2);
        }
}



