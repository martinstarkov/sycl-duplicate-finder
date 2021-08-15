#include "main.h"

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/imgproc/imgproc.hpp> // cv::cvtColor

#include "Timer.h"

#include <unordered_set>
#include <iostream>
#include <filesystem>
#include <vector>
#include <string>
#include <CL/sycl.hpp>

#include <stdio.h>
#include <stdlib.h>

#define LOG(x) { std::cout << x << std::endl; }

const char* wndname = "Square Detection Demo";

using Hash = std::int64_t;
using Path = std::filesystem::path;

enum class FileType {
    UNKNOWN = -1,
    PICTURE = 0,
    VIDEO = 1,
    HEIC = 2
};

FileType GetFileType(const Path& path) {
    auto extension{ path.extension().string() };
    if (extension == ".jpg" ||
        extension == ".png" ||
        extension == ".jpeg" ||
        extension == ".bmp") {
        return FileType::PICTURE;
    } else if (extension == ".mov" ||
               extension == ".mp4" ||
               extension == ".wmv" ||
               extension == ".gif" ||
               extension == ".avi") {
        return FileType::VIDEO;
    } else if (extension == ".heic" ||
               extension == ".heif") {
        return FileType::HEIC;
    } else {
        return FileType::UNKNOWN;
    }
}

struct File {
    File(Hash hash, const Path& path) : hash{ hash }, path{ path } {}
    Hash hash{ 0 };
    Path path;
    bool operator==(const File& other) const {
        return path == other.path;
    }
    bool operator!=(const File& other) const {
        return !operator==(other);
    }
};

Hash GetAHash(const cv::Mat& resized_image) {
    Hash hash{ 0 };
    int i{ 0 }; // counts the current index in the hash
    // Cycle through every row
    std::size_t sum{ 0 };
    std::size_t pixels{ static_cast<std::size_t>(resized_image.rows * resized_image.cols) };
    for (int index = 0; index < pixels; ++index) {
        // Get pixel pointer for the row
        auto ptr = (std::uint8_t*)(resized_image.data + index);
        sum += *ptr;
    }
    std::size_t average{ sum / pixels };
    for (int y = 0; y < resized_image.rows; ++y) {
        // Get pixel pointer for the row
        auto ptr = (std::uint8_t*)(resized_image.data + y * resized_image.step);
        // Cycle through every column
        for (int x = 0; x < resized_image.cols; ++x) {
            if (ptr[x] > average) {
                hash |= (std::int64_t)1 << i;
            }
            i++; // increment hash index
        }
    }
    assert(hash != 0);
    return hash;
}

Hash GetDHash(const cv::Mat& resized_image) {
    assert(resized_image.rows == resized_image.cols - 1);
    Hash hash{ 0 };
    int i{ 0 }; // counts the current index in the hash
    // Cycle through every row 
    for (int y = 0; y < resized_image.rows; ++y) {
        // Get pixel pointer for the row
        auto ptr = (std::uint8_t*)(resized_image.data + y * resized_image.step);
        // Cycle through every column
        for (int x = 0; x < resized_image.cols - 1; ++x) {
            // If the next pixel is brighter, make the hash contain a 1, else keep it as 0
            if (ptr[x + 1] > ptr[x]) {
                hash |= (std::int64_t)1 << i;
            }
            i++; // increment hash index
        }
    }
    assert(hash != 0);
    return hash;
}

cv::Mat Resize(const cv::Mat& image, int width, int height) {
    cv::Mat resized_image;
    cv::resize(image, resized_image, { width, height });
    return resized_image;
}

cv::Mat ToGreyscale(const cv::Mat& image) {
    cv::Mat greyscale;
    cv::cvtColor(image, greyscale, cv::ColorConversionCodes::COLOR_BGR2GRAY);
    return greyscale;
}

cv::Mat GetImage(const Path& path) {
    switch (GetFileType(path)) {
        case FileType::PICTURE:
        {
            cv::Mat image{ cv::imread(path.generic_string()) };
            return image;
        }
        case FileType::VIDEO:
        {

            return {};
        }
        case FileType::HEIC:
        {

            return {};
        }
    }
    return {};
}


std::vector<File> GetHashes(const std::vector<Path>& files) {
    LOG("[Starting hashing of " << files.size() << " files]");
    std::vector<File> duplicates;
    std::size_t counter{ 0 };
    std::size_t file_fraction{ files.size() / static_cast<std::size_t>(100) };
    std::size_t loading{ 1 };
    for (const auto& path : files) {
        auto image{ GetImage(path) };

        auto greyscale{ ToGreyscale(image) };
        
        auto thumbnail{ Resize(greyscale, 9, 8) };
        
        auto hash{ GetDHash(thumbnail) };

        duplicates.emplace_back(File{ hash, path });

        counter += 1;
        if (counter >= file_fraction) {
            LOG("[" << loading << "% of files hashed]");
            loading += 1;
            counter = 0;
        }

    }
    LOG("[Finished hashing all files]");
    return duplicates;
}

std::vector<Path> GetFiles(const std::vector<const char*>& directories) {
    LOG("[Starting file search through " << directories.size() << " directories]");
    std::vector<Path> files;
    for (const auto& directory : directories) {
        // Cycle through each file in the given directory.
        for (const auto& file : std::filesystem::directory_iterator(directory)) {
            if (file.is_regular_file()) {
                files.emplace_back(std::filesystem::absolute(file.path()));
            }
        }
    }
    LOG("[Finished file search]");
    return files;
}

int GetHammingDistance(Hash hash1, Hash hash2) {
    Hash x{ hash1 ^ hash2 }; // XOR (get differences)
    int hamming_distance{ 0 };
    while (x > 0) {
        hamming_distance += x & 1; // (count differences)
        x >>= 1;
    }
    return hamming_distance;
}

struct DuplicatePair {
    DuplicatePair(const std::pair<File, File>& duplicates, int similarity) : duplicates{ duplicates }, similarity{ similarity } {}
    std::pair<File, File> duplicates;
    int similarity{ 0 };
    bool operator==(const DuplicatePair& other) const {
        return 
            (duplicates.first == other.duplicates.first && duplicates.second == other.duplicates.second && similarity == other.similarity) ||
            (duplicates.first == other.duplicates.second && duplicates.second == other.duplicates.first && similarity == other.similarity);
    }
    bool operator!=(const DuplicatePair& other) const {
        return !operator==(other);
    }
    struct HashFunction {
        std::size_t operator()(const DuplicatePair& pair) const {
            auto min{ std::min(pair.duplicates.first.hash, pair.duplicates.second.hash) };
            auto max{ std::max(pair.duplicates.first.hash, pair.duplicates.second.hash) };
            return min ^ (max << 1);
        }
    };
};

//using DuplicatePairs = std::unordered_set<DuplicatePair, DuplicatePair::HashFunction>;
using DuplicatePairs = std::vector<DuplicatePair>;

DuplicatePairs ProcessDuplicates(const std::vector<File>& hashes, int hamming_threshold) {
    LOG("[Starting duplicate search through " << hashes.size() << " hashes]");
    DuplicatePairs pairs;
    std::size_t counter{ 0 };
    std::size_t file_fraction{ hashes.size() / static_cast<std::size_t>(100) };
    std::size_t loading{ 1 };

    std::size_t hash_count{ hashes.size() };
    cl::sycl::range<1> hash_range{ hash_count };
    cl::sycl::buffer<DuplicatePair, 1> destination_b{ pairs, cl::sycl::property_list{ cl::sycl::property::buffer::use_host_ptr {} } };
    cl::sycl::buffer<File, 1> hash_b{ hashes.data(), hash_range };

    cl::sycl::queue queue;

    queue.submit([&](cl::sycl::handler& cgh) {
        auto destination_a = destination_b.template get_access<cl::sycl::access::mode::write>(cgh);
        auto hash_a = hash_b.template get_access<cl::sycl::access::mode::read>(cgh);
        auto kernel = [=](cl::sycl::id<1> i) {
            const auto& file1{ hash_a[i] };
            for (auto j{ 0 }; j < hash_count; ++j) {
                const auto& file2{ hash_a[j] };
                auto hamming_distance{ GetHammingDistance(file1.hash, file2.hash) };
                if (hamming_distance <= hamming_threshold) {
                    int similarity{ static_cast<int>(1.0 - static_cast<double>(hamming_distance) / (static_cast<double>(hamming_threshold) + 1.0) * 100.0) };
                    destination_a[i] = (DuplicatePair{ { file1, file2 }, similarity });
                }
            }
        };
        cgh.parallel_for<class find_similar_images>(hash_range, kernel);
    });

    //for (const auto& file1 : hashes) {
    //    for (const auto& file2 : hashes) {
    //        if (file1 != file2) {
    //            auto hamming_distance{ GetHammingDistance(file1.hash, file2.hash) };
    //            if (hamming_distance <= hamming_threshold) {
    //                int similarity{ static_cast<int>(1.0 - static_cast<double>(hamming_distance) / (static_cast<double>(hamming_threshold) + 1.0) * 100.0) };
    //                //pairs.emplace(std::pair<File, File>{ file1, file2 }, similarity);
    //            }
    //        }
    //    }
    //    counter += 1;
    //    if (counter >= file_fraction) {
    //        LOG("[" << loading << "% of duplicates processed]");
    //        loading += 1;
    //        counter = 0;
    //    }
    //}
    LOG("[Finished duplicate search]");
    return pairs;
}

int main(int argc, char** argv) {

    auto files{ GetFiles({ "../vizsla_154/", "../maltese_252/", "../vizsla_4048/" }) };
    //auto files{ GetFiles({ "../test/" }) };
    auto hashes{ GetHashes(files) };
    auto pairs{ ProcessDuplicates(hashes, 0) };
    
    cv::Size window{ 800, 400 };

    for (const auto& pair : pairs) {
        std::array<cv::Mat, 2> images{
            Resize(GetImage(pair.duplicates.first.path), window.width / 2, window.height),
            Resize(GetImage(pair.duplicates.second.path), window.width / 2, window.height)
        };
        cv::Mat concatenated;
        // Concatenate duplicates images into one big image
        cv::hconcat(images.data(), 2, concatenated);
        cv::imshow("Duplicate Finder", concatenated);
        cv::waitKey(0);
    }
    cv::destroyAllWindows();

}