#ifndef COLORMAP_H
#define COLORMAP_H

#include <stdint.h>

class ColorMap
{
public:

    /**
    * 将强度值转化为RGB颜色，便于显示
    * @param 拥有三个元素的数组
    * @param 强度值
    * @param 强度取值范围的下界
    * @param 强度取值范围的上界
    */
    static void jet(double *rgb, double x, double min, double max);
};

#endif // COLORMAP_H
