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

using namespace std;

// Константы
const int MIN_CSV_FIELDS = 34;        // Минимальное количество полей в CSV
const double PRICE_INTERVAL_SIZE = 5.0; // Размер ценового интервала в долларах
const int MAX_TOP_PUBLISHERS = 5;     // Количество топ издателей для вывода

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

// Функция для чтения и парсинга CSV файла
vector<Game> readCSV(const string& filename, int rank, int world_size) {
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
        // Распределяем строки между процессами
        if (line_number % world_size != rank) {
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
                     int rank, int world_size, ofstream& outfile) {
    
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
        
        // Вычисляем отношение положительных к отрицательным отзывам
        if (game.positive > 0 || game.negative > 0) {
            double ratio = 0.0;
            if (game.negative > 0) {
                ratio = static_cast<double>(game.positive) / static_cast<double>(game.negative);
            } else if (game.positive > 0) {
                ratio = static_cast<double>(game.positive);
            }
            stats.review_ratios.push_back(ratio);
        }
    }
    
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
                double sum = 0.0;
                for (double ratio : stats.review_ratios) {
                    sum += ratio;
                }
                stats.avg_review_ratio = sum / stats.review_ratios.size();
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
    
    if (rank == 0) {
        cout << "Запуск анализа Steam игр с использованием MPI\n";
        cout << "Количество процессов: " << world_size << "\n";
    }
    
    // Проверяем аргументы командной строки
    string csv_filename = "games.csv";
    if (argc > 1) {
        csv_filename = argv[1];
    }
    
    if (rank == 0) {
        cout << "Чтение файла: " << csv_filename << "\n";
    }
    
    // Засекаем время начала
    double start_time = MPI_Wtime();
    
    // Каждый процесс читает свою часть данных
    vector<Game> local_games = readCSV(csv_filename, rank, world_size);
    
    if (rank == 0) {
        cout << "Процесс " << rank << " прочитал " << local_games.size() 
             << " строк на узле " << node_name << "\n";
    }
    
    // Собираем данные со всех процессов на главном
    int local_count = local_games.size();
    int total_count = 0;
    MPI_Reduce(&local_count, &total_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        cout << "Всего прочитано игр: " << total_count << "\n";
        cout << "Начинаем анализ...\n";
    }
    
    // Открываем выходной файл только на главном процессе
    ofstream outfile;
    if (rank == 0) {
        outfile.open("analysis_results.txt");
        if (!outfile.is_open()) {
            cerr << "Ошибка: не удалось создать файл результатов\n";
            MPI_Finalize();
            return 1;
        }
        
        outfile << "Результаты анализа игр Steam\n";
        outfile << "========================================\n";
    }
    
    // Анализируем данные по каждой платформе
    analyzePlatform(local_games, "Windows", rank, world_size, outfile);
    MPI_Barrier(MPI_COMM_WORLD);
    
    analyzePlatform(local_games, "Mac", rank, world_size, outfile);
    MPI_Barrier(MPI_COMM_WORLD);
    
    analyzePlatform(local_games, "Linux", rank, world_size, outfile);
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (rank == 0) {
        outfile.close();
        cout << "Анализ завершен. Результаты сохранены в analysis_results.txt\n";
    }
    
    // Засекаем время окончания
    double end_time = MPI_Wtime();
    
    if (rank == 0) {
        cout << "Время выполнения: " << (end_time - start_time) << " секунд\n";
    }
    
    MPI_Finalize();
    return 0;
}
