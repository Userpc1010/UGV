#ifndef COSTMAPCOLORIZER_H
#define COSTMAPCOLORIZER_H

#include <QImage>
#include <array>
#include <vector>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <map>

struct RGBAColor {
    uint8_t r, g, b, a;
};

class CostmapColorizer {
private:
    std::array<RGBAColor, 256> color_palette_;

public:
    CostmapColorizer() {
        createColorPalette();
    }

    void createColorPalette() {
        // Инициализируем всю палитру полностью прозрачным черным
        for (int i = 0; i < 256; i++) {
            color_palette_[i] = {0, 0, 0, 0};
        }

        // Значение 0: светло-серый
        color_palette_[0] = {192, 192, 192, 255};

//        // Blue to red spectrum для значений 1-100
//        for (int i = 1; i <= 100; i++) {
//            uint8_t v = (255 * i) / 100;
//            color_palette_[i] = {v, 0, static_cast<uint8_t>(255 - v), 255};
//        }

        // Blue to red spectrum для значений 2-98 (центр в 50)
        for (int i = 1; i <= 100; i++) {
             int distance_from_center = abs(i - 49);  // расстояние от центра (50)
              uint8_t red_component = (255 * distance_from_center) / 49;  // чем дальше от 50, тем краснее
              uint8_t blue_component = 255 - red_component;  // чем дальше от 50, тем меньше синего

              color_palette_[i] = {red_component, 0, blue_component, 255};
        }

        // Illegal positive values (101-127) - green
        for (int i = 101; i <= 127; i++) {
            color_palette_[i] = {0, 255, 0, 255};
        }

        // Illegal negative values (128-252) - red/yellow gradient
        for (int i = 128; i <= 252; i++) {
            // Исправляем деление на 0 при i = 128
            int denominator = (252 - 128);
            if (denominator == 0) denominator = 1;
            color_palette_[i] = {255, static_cast<uint8_t>((255 * (i - 128)) / denominator), 0, 255};
        }

        // Inscribed obstacle (253) - cyan
        color_palette_[253] = {0, 255, 255, 255};

        // Lethal obstacle (254) - purple
        color_palette_[254] = {255, 0, 255, 255};

        // Unknown (-1, представлен как 255) - blueish greenish grayish
        color_palette_[255] = {0x70, 0x89, 0x86, 255};
    }

    void printCostmap(const uint8_t* costmap, int width = 400, int height = 400) {
        // Статистический анализ
        if (costmap == nullptr) {
            std::cout << "Costmap is null!\n";
            return;
        }

        int total_size = width * height;

        // Основная статистика
        uint8_t min_val = 255;
        uint8_t max_val = 0;
        double sum_val = 0.0;

        for (int i = 0; i < total_size; ++i) {
            uint8_t val = costmap[i];
            if (val < min_val) min_val = val;
            if (val > max_val) max_val = val;
            sum_val += val;
        }
        double mean_val = sum_val / total_size;

        // Медиана
        std::vector<uint8_t> sorted(total_size);
        std::copy(costmap, costmap + total_size, sorted.begin());
        std::sort(sorted.begin(), sorted.end());
        double median = (total_size % 2 == 0)
            ? (sorted[total_size/2 - 1] + sorted[total_size/2]) / 2.0
            : sorted[total_size/2];

        // Частотный анализ (топ-5 значений)
        std::map<uint8_t, int> frequency;
        for (int i = 0; i < total_size; ++i) {
            frequency[costmap[i]]++;
        }

        std::vector<std::pair<uint8_t, int>> freq_vec(frequency.begin(), frequency.end());
        std::sort(freq_vec.begin(), freq_vec.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });

        // Подсчет нулей и ненулевых значений
        int zero_count = std::count_if(costmap, costmap + total_size, [](uint8_t x) { return x == 0; });
        int nonzero_count = total_size - zero_count;

        // Вывод статистики
        std::cout << "=== COSTMAP STATISTICAL ANALYSIS ===\n";
        std::cout << "Dimensions: " << width << "x" << height << " (" << total_size << " elements)\n";
        std::cout << "Data type: uint8_t (0-255)\n";

        std::cout << "\n--- Value Distribution ---\n";
        std::cout << "Min value: " << static_cast<int>(min_val) << "\n";
        std::cout << "Max value: " << static_cast<int>(max_val) << "\n";
        std::cout << "Mean: " << std::fixed << std::setprecision(2) << mean_val << "\n";
        std::cout << "Median: " << median << "\n";

        std::cout << "\n--- Value Categories ---\n";
        std::cout << "Zero values: " << zero_count << " (" << (zero_count * 100.0 / total_size) << "%)\n";
        std::cout << "Non-zero values: " << nonzero_count << " (" << (nonzero_count * 100.0 / total_size) << "%)\n";

        std::cout << "\n--- Most Frequent Values (Top 10) ---\n";
        int count = 0;
        for (const auto& [value, freq] : freq_vec) {
            if (count++ >= 10) break;
            std::cout << "Value " << std::setw(3) << static_cast<int>(value) << ": " << std::setw(6) << freq
                      << " times (" << std::setw(5) << std::fixed << std::setprecision(1)
                      << (freq * 100.0 / total_size) << "%)\n";
        }

        // Визуализация распределения (гистограмма)
        std::cout << "\n--- Value Histogram ---\n";
        const int histogram_bins = std::min(static_cast<int>(max_val - min_val + 1), 50);

        if (histogram_bins <= 50) {
            // Подробная гистограмма для небольших диапазонов
            for (uint8_t i = min_val; i <= max_val; i++) {
                int count = frequency[i];
                if (count > 0) {
                    std::cout << std::setw(3) << static_cast<int>(i) << ": "
                              << std::string(count * 60 / total_size, '#')
                              << " (" << count << ")\n";
                }
            }
        } else {
            // Группированная гистограмма для больших диапазонов
            int bin_size = (max_val - min_val + 1) / 20 + 1;
            for (int bin = 0; bin < 20; bin++) {
                uint8_t start = min_val + bin * bin_size;
                uint8_t end = std::min((uint8_t)(min_val + (bin + 1) * bin_size - 1), max_val);

                int bin_count = 0;
                for (uint8_t val = start; val <= end; val++) {
                    bin_count += frequency[val];
                }

                if (bin_count > 0) {
                    std::cout << std::setw(3) << static_cast<int>(start) << "-"
                              << std::setw(3) << static_cast<int>(end) << ": "
                              << std::string(bin_count * 60 / total_size, '#')
                              << " (" << bin_count << ")\n";
                }
            }
        }

        // Дополнительная информация о распределении
        std::cout << "\n--- Range Analysis ---\n";
        int low_range = 0, mid_range = 0, high_range = 0;
        for (int i = 0; i < total_size; ++i) {
            uint8_t val = costmap[i];
            if (val <= 85) low_range++;           // 0-85
            else if (val <= 170) mid_range++;     // 86-170
            else high_range++;                    // 171-255
        }

        std::cout << "Low range (0-85): " << low_range << " (" << (low_range * 100.0 / total_size) << "%)\n";
        std::cout << "Mid range (86-170): " << mid_range << " (" << (mid_range * 100.0 / total_size) << "%)\n";
        std::cout << "High range (171-255): " << high_range << " (" << (high_range * 100.0 / total_size) << "%)\n";

        std::cout << "\n=== END OF ANALYSIS ===\n\n";
    }

    // Версия для работы с сырыми массивами
    QImage colorizeCostmap(const uint8_t* costmap_data, int width, int height) {
        QImage image(width, height, QImage::Format_RGBA8888);

        // Проверка на nullptr
        if (!costmap_data) {
            std::cerr << "Warning: costmap_data is null in colorizeCostmap\n";
            return image;
        }

        // Проверка размеров
        if (width <= 0 || height <= 0) {
            std::cerr << "Warning: invalid dimensions in colorizeCostmap: "
                      << width << "x" << height << "\n";
            return image;
        }

        for (int y = 0; y < height; y++) {
            uint8_t* scanLine = image.scanLine(y);
            for (int x = 0; x < width; x++) {
                int costmap_index = y * width + x;
                RGBAColor color = getColorForValue(costmap_data[costmap_index]);
                int pixel_offset = x * 4;
                scanLine[pixel_offset + 0] = color.r;
                scanLine[pixel_offset + 1] = color.g;
                scanLine[pixel_offset + 2] = color.b;
                scanLine[pixel_offset + 3] = color.a;
            }
        }
        return image;
    }

    // Получение цвета для отдельного значения
    QRgb getQRgbForValue(int cost_value) {
        RGBAColor color = getColorForValue(cost_value);
        return qRgba(color.r, color.g, color.b, color.a);
    }

private:
    RGBAColor getColorForValue(int cost_value) {
        // Исправление: проверяем границы и возвращаем цвет
        if (cost_value < 0 || cost_value >= 256) {
            // Возвращаем прозрачный черный для некорректных значений
            return {0, 0, 0, 0};
        }
        return color_palette_[cost_value];
    }

    // Также добавьте безопасную версию для uint8_t
    RGBAColor getColorForValue(uint8_t cost_value) {
        return color_palette_[cost_value];
    }
};

#endif // COSTMAPCOLORIZER_H
