#include <mpi.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <cmath>
#include <iomanip>

// Поддержка CUDA (опционально)
#ifdef USE_CUDA
#include "cuda_accelerator.h"
#endif

using namespace std;

// Константы
const int MIN_CSV_FIELDS = 34;        // Минимальное количество полей в CSV
const double PRICE_INTERVAL_SIZE = 5.0; // Размер ценового интервала в долларах
const int MAX_TOP_PUBLISHERS = 5;     // Количество топ издателей для вывода
const int GPU_WEIGHT = 3;              // Вес GPU узла (в 3 раза мощнее CPU)
const int CPU_WEIGHT = 1;              // Вес CPU узла

// Структура для хранения данных об игре
struct Game {
    string app_id;
    string name;
    double price;
    int dlc_count;
    bool windows;
    bool mac;
    bool linux;
    int positive;
    int negative;
    string publisher;
};

// Структура для статистики издателя
struct PublisherStats {
    string name;
    int game_count;
    int total_dlc;
    double avg_review_ratio;
    vector<double> review_ratios;
    
    PublisherStats() : game_count(0), total_dlc(0), avg_review_ratio(0.0) {}
};

// Функция для парсинга CSV строки с учетом кавычек
vector<string> parseCSVLine(const string& line) {
    vector<string> result;
    string field;
    bool inQuotes = false;
    
    for (size_t i = 0; i < line.length(); i++) {
        char c = line[i];
        
        if (c == '"') {
            inQuotes = !inQuotes;
        } else if (c == ',' && !inQuotes) {
            result.push_back(field);
            field.clear();
        } else {
            field += c;
        }
    }
    result.push_back(field);
    
    return result;
}

// Функция для конвертации строки в bool
bool stringToBool(const string& str) {
    return (str == "True" || str == "true" || str == "1");
}

// Функция для безопасного преобразования строки в double
double safeStod(const string& str) {
    try {
        if (str.empty()) return 0.0;
        return stod(str);
    } catch (...) {
        return 0.0;
    }
}

// Функция для безопасного преобразования строки в int
int safeStoi(const string& str) {
    try {
        if (str.empty()) return 0;
        return stoi(str);
    } catch (...) {
        return 0;
    }
}

// Функция для определения типа узла (GPU или CPU)
bool isGPUNode(const char* node_name) {
    // Проверяем по имени узла
    string name(node_name);
    return (name.find("gpu") != string::npos || name.find("GPU") != string::npos);
}

// Функция для получения веса узла
int getNodeWeight(const char* node_name) {
    return isGPUNode(node_name) ? GPU_WEIGHT : CPU_WEIGHT;
}

// Функция для вычисления взвешенного распределения строк
// Возвращает true если текущий процесс должен обработать данную строку
bool shouldProcessLine(int line_number, int rank, int world_size, 
                       const vector<int>& rank_weights, int total_weight) {
    // Проверка на некорректные входные данные
    if (total_weight <= 0) {
        return false;
    }
    
    // Вычисляем к какому процессу относится данная строка на основе весов
    int weight_position = line_number % total_weight;
    int cumulative_weight = 0;
    
    for (int r = 0; r < world_size; r++) {
        cumulative_weight += rank_weights[r];
        if (weight_position < cumulative_weight) {
            return (r == rank);
        }
    }
    
    return false;
}

// Функция для чтения и парсинга CSV файла с учетом весов узлов
vector<Game> readCSV(const string& filename, int rank, int world_size,
                     const vector<int>& rank_weights, int total_weight) {
    vector<Game> games;
    ifstream file(filename);
    
    if (!file.is_open()) {
        if (rank == 0) {
            cerr << "Ошибка: не удалось открыть файл " << filename << endl;
        }
        return games;
    }
    
    string line;
    // Пропускаем заголовок
    getline(file, line);
    
    int line_number = 0;
    while (getline(file, line)) {
        // Распределяем строки между процессами с учетом весов
        if (!shouldProcessLine(line_number, rank, world_size, rank_weights, total_weight)) {
            line_number++;
            continue;
        }
        line_number++;
        
        vector<string> fields = parseCSVLine(line);
        
        // Проверяем, что достаточно полей
        if (fields.size() < MIN_CSV_FIELDS) continue;
        
        Game game;
        game.app_id = fields[0];
        game.name = fields[1];
        game.price = safeStod(fields[6]);
        game.dlc_count = safeStoi(fields[8]);
        game.windows = stringToBool(fields[17]);
        game.mac = stringToBool(fields[18]);
        game.linux = stringToBool(fields[19]);
        game.positive = safeStoi(fields[23]);
        game.negative = safeStoi(fields[24]);
        game.publisher = fields[33];
        
        games.push_back(game);
    }
    
    file.close();
    return games;
}

// Функция для анализа данных по платформе
void analyzePlatform(const vector<Game>& games, const string& platform_name, 
                     int rank, int world_size, ofstream& outfile, ofstream& csvfile, 
                     bool use_cuda) {
    
    // Фильтруем игры по платформе
    vector<Game> platform_games;
    for (const auto& game : games) {
        bool include = false;
        if (platform_name == "Windows" && game.windows) include = true;
        else if (platform_name == "Mac" && game.mac) include = true;
        else if (platform_name == "Linux" && game.linux) include = true;
        
        if (include) {
            platform_games.push_back(game);
        }
    }
    
    // Группируем по ценовым интервалам ($5)
    map<int, map<string, PublisherStats>> price_groups;
    
    for (const auto& game : platform_games) {
        int price_interval = static_cast<int>(game.price / PRICE_INTERVAL_SIZE);
        string publisher = game.publisher;
        
        if (publisher.empty()) continue;
        
        auto& stats = price_groups[price_interval][publisher];
        stats.name = publisher;
        stats.game_count++;
        stats.total_dlc += game.dlc_count;
        
        // Сохраняем данные для потенциальной обработки на GPU
        if (game.positive > 0 || game.negative > 0) {
            // Для CPU-версии вычисляем сразу
            if (!use_cuda) {
                double ratio = 0.0;
                if (game.negative > 0) {
                    ratio = static_cast<double>(game.positive) / static_cast<double>(game.negative);
                } else if (game.positive > 0) {
                    ratio = static_cast<double>(game.positive);
                }
                stats.review_ratios.push_back(ratio);
            } else {
                // Для GPU-версии сохраняем исходные данные
                stats.review_ratios.push_back(game.positive); // временно используем для positive
                stats.review_ratios.push_back(game.negative);  // следующий элемент - negative
            }
        }
    }
    
#ifdef USE_CUDA
    // Если используется CUDA, обрабатываем review ratios на GPU
    if (use_cuda) {
        for (auto& price_group : price_groups) {
            for (auto& pub_pair : price_group.second) {
                auto& stats = pub_pair.second;
                if (stats.review_ratios.size() >= 2) {
                    int count = stats.review_ratios.size() / 2;
                    vector<int> positive_vals(count);
                    vector<int> negative_vals(count);
                    vector<double> ratios(count);
                    
                    // Разделяем positive и negative
                    for (int i = 0; i < count; i++) {
                        positive_vals[i] = static_cast<int>(stats.review_ratios[i * 2]);
                        negative_vals[i] = static_cast<int>(stats.review_ratios[i * 2 + 1]);
                    }
                    
                    // Вычисляем на GPU
                    if (cuda_calculate_review_ratios(positive_vals.data(), negative_vals.data(), 
                                                     ratios.data(), count)) {
                        stats.review_ratios = ratios;
                    } else {
                        // Fallback на CPU если CUDA не сработала
                        stats.review_ratios.clear();
                        for (int i = 0; i < count; i++) {
                            double ratio = 0.0;
                            if (negative_vals[i] > 0) {
                                ratio = static_cast<double>(positive_vals[i]) / static_cast<double>(negative_vals[i]);
                            } else if (positive_vals[i] > 0) {
                                ratio = static_cast<double>(positive_vals[i]);
                            }
                            stats.review_ratios.push_back(ratio);
                        }
                    }
                }
            }
        }
    }
#endif
    
    // Для каждого ценового интервала находим топ-5 издателей
    if (rank == 0) {
        outfile << "\n========================================\n";
        outfile << "Платформа: " << platform_name << "\n";
        outfile << "========================================\n";
    }
    
    for (auto& price_group : price_groups) {
        int interval = price_group.first;
        auto& publishers = price_group.second;
        
        // Вычисляем среднее отношение отзывов
        for (auto& pub_pair : publishers) {
            auto& stats = pub_pair.second;
            if (!stats.review_ratios.empty()) {
#ifdef USE_CUDA
                // Используем CUDA для вычисления среднего на GPU узлах
                if (use_cuda && stats.review_ratios.size() > 10) {
                    stats.avg_review_ratio = cuda_calculate_average(
                        stats.review_ratios.data(), stats.review_ratios.size()
                    );
                } else {
#endif
                    // CPU версия
                    double sum = 0.0;
                    for (double ratio : stats.review_ratios) {
                        sum += ratio;
                    }
                    stats.avg_review_ratio = sum / stats.review_ratios.size();
#ifdef USE_CUDA
                }
#endif
            }
        }
        
        // Создаем вектор для сортировки
        vector<PublisherStats> sorted_publishers;
        for (auto& pub_pair : publishers) {
            sorted_publishers.push_back(pub_pair.second);
        }
        
        // Сортируем по отношению DLC/игры (убывание)
        sort(sorted_publishers.begin(), sorted_publishers.end(), 
             [](const PublisherStats& a, const PublisherStats& b) {
                 double ratio_a = a.game_count > 0 ? 
                     static_cast<double>(a.total_dlc) / a.game_count : 0.0;
                 double ratio_b = b.game_count > 0 ? 
                     static_cast<double>(b.total_dlc) / b.game_count : 0.0;
                 return ratio_a > ratio_b;
             });
        
        // Выводим топ-5
        if (rank == 0 && !sorted_publishers.empty()) {
            outfile << "\nЦеновой интервал: $" << (interval * static_cast<int>(PRICE_INTERVAL_SIZE)) 
                    << " - $" << ((interval + 1) * static_cast<int>(PRICE_INTERVAL_SIZE)) << "\n";
            outfile << "----------------------------------------\n";
            
            int count = 0;
            for (const auto& stats : sorted_publishers) {
                if (count >= MAX_TOP_PUBLISHERS) break;
                
                double dlc_ratio = stats.game_count > 0 ? 
                    static_cast<double>(stats.total_dlc) / stats.game_count : 0.0;
                
                outfile << "Издатель: " << stats.name << "\n";
                outfile << "  Количество игр: " << stats.game_count << "\n";
                outfile << "  Отношение DLC/игра: " << fixed << setprecision(2) 
                        << dlc_ratio << "\n";
                outfile << "  Среднее отношение положительных к отрицательным отзывам: " 
                        << fixed << setprecision(2) << stats.avg_review_ratio << "\n";
                outfile << "\n";
                
                // Выводим в CSV формате
                csvfile << platform_name << ","
                       << (interval * static_cast<int>(PRICE_INTERVAL_SIZE)) << ","
                       << ((interval + 1) * static_cast<int>(PRICE_INTERVAL_SIZE)) << ","
                       << "\"" << stats.name << "\","
                       << stats.game_count << ","
                       << fixed << setprecision(2) << dlc_ratio << ","
                       << fixed << setprecision(2) << stats.avg_review_ratio << "\n";
                
                count++;
            }
        }
    }
}

int main(int argc, char** argv) {
    // Инициализация MPI
    MPI_Init(&argc, &argv);
    
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    char node_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(node_name, &name_len);
    
    // Определяем тип узла и его вес
    bool has_gpu = isGPUNode(node_name);
    int node_weight = getNodeWeight(node_name);
    
    // Инициализация CUDA на GPU узлах
    bool use_cuda = false;
#ifdef USE_CUDA
    if (has_gpu) {
        use_cuda = cuda_initialize();
        if (use_cuda) {
            cout << "Процесс " << rank << ": CUDA инициализирована успешно\n";
        } else {
            cout << "Процесс " << rank << ": CUDA недоступна, используется CPU режим\n";
        }
    }
#endif
    
    // Собираем информацию о весах всех процессов
    vector<int> all_weights(world_size);
    int mpi_result = MPI_Allgather(&node_weight, 1, MPI_INT, all_weights.data(), 1, MPI_INT, MPI_COMM_WORLD);
    
    if (mpi_result != MPI_SUCCESS) {
        if (rank == 0) {
            cerr << "Ошибка: MPI_Allgather вернул код ошибки " << mpi_result << "\n";
        }
        MPI_Finalize();
        return 1;
    }
    
    // Вычисляем общий вес
    int total_weight = 0;
    for (int w : all_weights) {
        total_weight += w;
    }
    
    if (rank == 0) {
        cout << "Запуск анализа Steam игр с использованием MPI и GPU ускорения\n";
        cout << "Количество процессов: " << world_size << "\n";
        cout << "Общий вычислительный вес: " << total_weight << "\n";
        cout << "\nРаспределение по узлам:\n";
    }
    
    // Все процессы выводят информацию о себе
    for (int i = 0; i < world_size; i++) {
        if (rank == i) {
            cout << "  Процесс " << rank << " на узле " << node_name 
                 << " (тип: " << (has_gpu ? "GPU" : "CPU") 
                 << ", вес: " << node_weight << ")\n";
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    // Проверяем аргументы командной строки
    string csv_filename = "games.csv";
    if (argc > 1) {
        csv_filename = argv[1];
    }
    
    if (rank == 0) {
        cout << "\nЧтение файла: " << csv_filename << "\n";
    }
    
    // Засекаем время начала
    double start_time = MPI_Wtime();
    
    // Каждый процесс читает свою часть данных с учетом весов
    vector<Game> local_games = readCSV(csv_filename, rank, world_size, all_weights, total_weight);
    
    if (rank == 0) {
        cout << "Данные распределены с учетом мощности узлов:\n";
    }
    
    // Все процессы выводят информацию о своей загрузке
    for (int i = 0; i < world_size; i++) {
        if (rank == i) {
            cout << "  Процесс " << rank << " (" << (has_gpu ? "GPU" : "CPU") 
                 << ") обработал " << local_games.size() << " строк\n";
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    // Собираем данные со всех процессов на главном
    int local_count = local_games.size();
    int total_count = 0;
    MPI_Reduce(&local_count, &total_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        cout << "\nВсего прочитано игр: " << total_count << "\n";
        cout << "Начинаем анализ...\n";
    }
    
    // Открываем выходные файлы только на главном процессе
    ofstream outfile;
    ofstream csvfile;
    if (rank == 0) {
        outfile.open("analysis_results.txt");
        if (!outfile.is_open()) {
            cerr << "Ошибка: не удалось создать файл результатов\n";
            MPI_Finalize();
            return 1;
        }
        
        csvfile.open("analysis_results.csv");
        if (!csvfile.is_open()) {
            cerr << "Ошибка: не удалось создать CSV файл результатов\n";
            MPI_Finalize();
            return 1;
        }
        
        outfile << "Результаты анализа игр Steam\n";
        outfile << "========================================\n";
        
        // Записываем заголовок CSV
        csvfile << "Платформа,Цена_мин,Цена_макс,Издатель,Количество_игр,DLC_на_игру,Отношение_отзывов\n";
    }
    
    // Анализируем данные по каждой платформе
    analyzePlatform(local_games, "Windows", rank, world_size, outfile, csvfile, use_cuda);
    MPI_Barrier(MPI_COMM_WORLD);
    
    analyzePlatform(local_games, "Mac", rank, world_size, outfile, csvfile, use_cuda);
    MPI_Barrier(MPI_COMM_WORLD);
    
    analyzePlatform(local_games, "Linux", rank, world_size, outfile, csvfile, use_cuda);
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (rank == 0) {
        outfile.close();
        csvfile.close();
        cout << "Анализ завершен. Результаты сохранены в analysis_results.txt и analysis_results.csv\n";
    }
    
    // Засекаем время окончания
    double end_time = MPI_Wtime();
    
    if (rank == 0) {
        cout << "Время выполнения: " << (end_time - start_time) << " секунд\n";
    }
    
#ifdef USE_CUDA
    // Очистка CUDA ресурсов на GPU узлах
    if (use_cuda) {
        cuda_cleanup();
    }
#endif
    
    MPI_Finalize();
    return 0;
}
