#ifndef MAP_CONFIG_H
#define MAP_CONFIG_H


#define MAP_SIZE 400                               // side length (cells)
#define Map_X 400
#define Map_Y 400
#define MAP_CELLS         (MAP_SIZE * MAP_SIZE)    // 160000
#define MAP_LAYERS        7

#define MAP_CENTER        (MAP_SIZE / 2)           // 200
#define MAP_LAST          (MAP_SIZE - 1)           // 399
#define MAP_CENTER_F      200.0f                   // float-версия для ядер
#define MAP_SIZE_F        400.0f
#define MAP_HALF          (MAP_SIZE / 2)           // для упаковки heightmap

// Индексы слоёв (совпадают с LayerNames в EMC)
#define L_HEIGHT          0
#define L_VARIANCE        1
#define L_IS_VALID        2
#define L_TRAVERS         3
#define L_TIME            4
#define L_UPPER_BOUND     5
#define L_IS_UPPER        6

// Утилита: плоский индекс ячейки (x — колонка, y — строка)
#define IDX(x, y)         ((y) * MAP_SIZE + (x))

// Параметры, которые были размазаны по коду
#define MAP_RESOLUTION    0.1f                     // м/ячейка
#define TIME_INTERVAL     0.1f                     // сек (10 Гц)
#define SLOPE_CELL_SIZE   100.0f                   // мм на ячейку в heightmap_to_slope

#define Map_scale 10.0f

#define RS2_CUDA_THREADS_PER_BLOCK 128

#endif // MAP_CONFIG_H
