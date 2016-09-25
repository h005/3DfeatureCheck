#include "reverseface.h"
#include <stdlib.h>
#include <stdio.h>

ReverseFace::ReverseFace()
{

}

ReverseFace::ReverseFace(std::vector<int> &indices)
{
    shrink(indices);
    std::cout << "shrinkd done..." << std::endl;

//    freopen("/home/h005/Documents/vpDataSet/bigben/imgs/model/bigben.rev","w",stdout);

//    for(int i=0;i<indices.size();i+=3)
//        printf("%d %d %d\n",indices[i],indices[i+1],indices[i+2]);

//    fclose(stdout);

    NUM_FACE = indices.size() / 3;
    // 删除边出现多余三次的面
    setRelation(indices);

    sz = new int[NUM_FACE];
    memset(sz,0,sizeof(int) * NUM_FACE);

    id = new int[NUM_FACE];
    for(int i=0;i<NUM_FACE;i++)
        id[i] = i;
}

ReverseFace::~ReverseFace()
{

}

int *ReverseFace::reverseFace(std::vector<int> &indices, std::vector<int> &cs, std::vector<std::set<int> > &cate)
{


//    reverse(indices);
    // set cateSet and id
    char visit[NUM_FACE];
    memset(visit,0,sizeof(char)*NUM_FACE);

    // clear cate
    cate.clear();

    int count = 0;
    // visit
    for(int i=0;i<NUM_FACE;i++)
    {
        if(visit[i])
            continue;
        std::set<int> tmpSet;
        visit[i] = 1;
        id[i] = count;
        tmpSet.insert(i);
        dfsReverse(indices,visit,i,count,tmpSet);
        cs.push_back(count);
        count++;
        cate.push_back(tmpSet);
    }

    return id;
}

void ReverseFace::shrink(std::vector<int> &indices)
{
    NUM_FACE = indices.size() / 3;
    std::set<QString> indSet;
    for(int i=0;i<NUM_FACE;i++)
    {
        int arr[3] = {indices[i*3],indices[i*3+1],indices[i*3+2]};
        std::sort(arr,arr+3);
        QString ss = "";
        QString ss1,ss2,ss3;
        ss.setNum(arr[0]);
        ss += " ";
        ss2.setNum(arr[1]);
        ss2 += " ";
        ss3.setNum(arr[2]);
        ss += ss2;
        ss += ss3;
        std::set<QString>::iterator it = indSet.find(ss);
        if(it != indSet.end())
        {
            indices.erase(indices.begin()+i*3,indices.begin() + i*3+3);
            i--;
            NUM_FACE = indices.size() / 3;
            continue;
        }
        indSet.insert(ss);
        NUM_FACE = indices.size() / 3;
    }

}

bool ReverseFace::isSameFace(std::vector<int> &indices, int i1, int i2)
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

void ReverseFace::setRelation(std::vector<int> &indices)
{
    adjt.clear();
    // initial adjt
    for(int i=0;i<NUM_FACE;i++)
    {
        std::vector<int> tmp;
        adjt.push_back(tmp);
    }

    char visit[NUM_FACE];
    memset(visit,0,sizeof(char) * NUM_FACE );

    adjTable.clear();

    // edgeID and appear times
    // it is used for check ,if appear times more than twice , assert!
    std::map< std::pair<int,int>, int > edges;
    edges.clear();

    for(int i=0;i<NUM_FACE;i++)
    {
        int tmpArray[3] = {indices[i*3], indices[i*3+1],indices[i*3+2]};
        std::sort(tmpArray,tmpArray+3);
        std::vector< std::pair<int,int> > pairs;
        pairs.push_back(std::make_pair<int,int>(tmpArray[0],tmpArray[1]));
        pairs.push_back(std::make_pair<int,int>(tmpArray[0],tmpArray[2]));
        pairs.push_back(std::make_pair<int,int>(tmpArray[1],tmpArray[2]));

        // attention 大本钟模型，存在一条边被多个三角形共用的情况！
        // 解决方案: 直接删除后面共用边的三角形
        int flag = 0;
        for(int j=0;j<3;j++)
        {
            std::map< std::pair<int,int>, int>::iterator it = edges.find(pairs[j]);
            if(it != edges.end())
            {
                it->second++;
                if(it->second == 3)
                {
                    std::cout << "debug pairs "<< pairs[j].first << " " << pairs[j].second << std::endl;
                    flag = 1;
                }
//                Q_ASSERT(it->second < 3);
            }
            else
                edges.insert(std::make_pair<std::pair<int,int>,int>(pairs[j],1));
        }
        if(flag)
        {
            for(int j=0;j<3;j++)
            {
                std::map< std::pair<int,int>, int>::iterator it = edges.find(pairs[j]);
                if( it->second == 1)
                    edges.erase(it);
                else
                    it->second--;
            }
            indices.erase(indices.begin() + i * 3, indices.begin() + i * 3 + 3);
            i--;
            NUM_FACE--;

        }
        else
        {
            for(int j=0;j<3;j++)
            {
                std::map< std::pair<int,int>, int>::iterator it = adjTable.find(pairs[j]);
                if(it != adjTable.end())
                {
                    adjt[it->second].push_back(i);
                    adjt[i].push_back(it->second);
                    adjTable.erase(it);
                }
                else
                    adjTable.insert( std::make_pair< std::pair<int,int>, int>(pairs[j],i));
            }
        }
    }

}

void ReverseFace::reverse(std::vector<int> &indices)
{
    char reverse_flag[NUM_FACE];
    memset(reverse_flag,0,sizeof(NUM_FACE));

    for(int i=0;i<NUM_FACE;i++)
    {
        if(reverse_flag[i])
            continue;
        reverse_flag[i] = 1;
        for(int j=0;j<adjt[i].size();j++)
        {
            if(reverse_flag[adjt[i][j]])
                continue;
            reverse_flag[adjt[i][j]] = 1;
            if(checkOrder(indices,i,adjt[i][j]))
            {
                // reverse
                int tmp = indices[adjt[i][j]*3];
                indices[adjt[i][j]*3] = indices[adjt[i][j]*3 + 2];
                indices[adjt[i][j]*3 + 2] = tmp;
            }
        }
    }

}

bool ReverseFace::dfsReverse(std::vector<int> &indices,char *visit,int ind,int count, std::set<int> &tmpSet)
{
    int flag = 0;
    for(int i=0;i<adjt[ind].size();i++)
    {
        // if not visit
        if(!visit[adjt[ind][i]])
        {
            visit[adjt[ind][i]] = 1;
            flag = 1;
            id[adjt[ind][i]] = count;
            tmpSet.insert(adjt[ind][i]);
            if(checkOrder(indices,ind,adjt[ind][i]))
            {
                // reverse
                int tmp = indices[adjt[ind][i]*3];
                indices[adjt[ind][i]*3] = indices[adjt[ind][i]*3+2];
                indices[adjt[ind][i]*3+2] = tmp;
            }
            dfsReverse(indices,visit,adjt[ind][i],count,tmpSet);
        }
        else
            continue;
    }
    if(flag == 0)
        return 0;
    else
        return 1;
}

bool ReverseFace::checkOrder(std::vector<int> &indices, int ind1, int ind2)
{
    std::set< std::pair<int,int> > pair1;
    std::vector< std::pair<int,int> > pair2;

    std::pair<int,int> pair = std::make_pair<int,int>(indices[ind1*3],indices[ind1*3+1]);
    pair1.insert(pair);
    pair = std::make_pair<int,int>(indices[ind1*3+1],indices[ind1*3+2]);
    pair1.insert(pair);
    pair = std::make_pair<int,int>(indices[ind1*3+2],indices[ind1*3]);
    pair1.insert(pair);

    pair = std::make_pair<int,int>(indices[ind2*3],indices[ind2*3+1]);
    pair2.push_back(pair);
    pair = std::make_pair<int,int>(indices[ind2*3+1],indices[ind2*3+2]);
    pair2.push_back(pair);
    pair = std::make_pair<int,int>(indices[ind2*3+2],indices[ind2*3]);
    pair2.push_back(pair);

    for(int i=0;i<3;i++)
    {
        std::set< std::pair<int,int> >::iterator it = pair1.find(pair2[i]);
        if(it != pair1.end())
            return true;
    }
    return false;
}

int ReverseFace::dfs(char *visit, int ind,int count)
{
    int flag = 0;
    for(int i=0;i<adjt[ind].size();i++)
    {
        // if not visit
        if(!visit[adjt[ind][i]])
        {
            visit[adjt[ind][i]] = 1;
            flag = 1;
            id[adjt[ind][i]] = count;
            dfs(visit,adjt[ind][i],count);
        }
        else
            continue;
    }
    if(flag == 0)
        return 0;
    else
        return 1;
}

