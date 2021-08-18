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

// TODO: Figure out why hash for DHash and AHash is negative.

#define LOG(x) { std::cout << x << std::endl; }

const char* wndname = "Square Detection Demo";

using Hash = std::uint64_t;
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
                hash |= (Hash)1 << i;
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
                hash |= (Hash)1 << i;
            }
            i++; // increment hash index
        }
    }
    assert(hash != 0);
    return hash;
}

inline std::uint64_t GetLinearIndex(std::size_t hash_count, std::uint64_t i, std::uint64_t j) {
    return hash_count * (hash_count - 1) / 2 - (hash_count - j) * ((hash_count - j) - 1) / 2 + i - j - 1;
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


std::vector<Hash> GetHashes(const std::vector<Path>& files) {
    std::size_t file_count{ files.size() };
    LOG("[Starting hashing of " << file_count << " files]");

    std::vector<Hash> hashes;
    hashes.resize(file_count, 0);

    std::size_t counter{ 0 };
    std::size_t file_fraction{ file_count / static_cast<std::size_t>(100) };
    std::size_t loading{ 1 };

    for (auto i{ 0 }; i < file_count; ++i) {
        const auto& path{ files[i] };
        auto image{ GetImage(path) };

        auto greyscale{ ToGreyscale(image) };
        
        auto thumbnail{ Resize(greyscale, 8, 8) };
        
        hashes[i] = GetAHash(thumbnail);

        counter += 1;
        if (counter >= file_fraction) {
            LOG("[" << loading << "% of files hashed]");
            loading += 1;
            counter = 0;
        }

    }
    LOG("[Finished hashing all files]");
    return hashes;
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

inline std::uint64_t GetHammingDistance(Hash hash1, Hash hash2) {
    return sycl::popcount(hash1 ^ hash2); // XOR (get number of 1 bits in the difference of 1 and 2).
}

using Pair = std::uint8_t;
using PairContainer = std::vector<std::uint8_t>;

PairContainer ProcessDuplicates(std::vector<Hash>& hashes, int hamming_threshold) {
    std::size_t hash_count{ hashes.size() };
    LOG("[Starting duplicate search through " << hash_count << " hashes]");

    std::size_t counter{ 0 };
    std::size_t file_fraction{ hashes.size() / static_cast<std::size_t>(100) };
    std::size_t loading{ 1 };
    cl::sycl::property_list properties{ cl::sycl::property::buffer::use_host_ptr {} };

    cl::sycl::range<1> hash_range{ hash_count };

    std::size_t pair_count{ hash_count * (hash_count - 1) / 2 };

    cl::sycl::range<1> destination_range{ pair_count };
    
    PairContainer pairs;
    pairs.resize(pair_count, 0);
    {
    cl::sycl::buffer<Hash, 1> hash_b{ hashes, properties };
    cl::sycl::buffer<Pair, 1> destination_b{ pairs, properties };

    cl::sycl::queue queue;

    LOG("PreHashes [i=4, j=1]: [" << hashes[4] << ", " << hashes[1] << "]");
    auto debug_index = GetLinearIndex(hash_count, 4, 1);
    LOG("PrePair is: " << debug_index << ":" << unsigned(pairs[debug_index]));
    LOG("PreHamming: " << GetHammingDistance(hashes[4], hashes[1]));

    queue.submit([&](cl::sycl::handler& cgh) {
        auto destination_a = destination_b.template get_access<cl::sycl::access::mode::read_write>(cgh);
        auto hash_a = hash_b.template get_access<cl::sycl::access::mode::read>(cgh);
        auto os = sycl::stream{ 50000, 4000, cgh };
        auto kernel = [=](cl::sycl::id<1> id) {
            auto i{ id.get(0) };
            for (auto j{ 0 }; j < i; ++j) {
                auto index = GetLinearIndex(hash_count, i, j);
                auto hamming_distance{ GetHammingDistance(hash_a[i], hash_a[j]) };
                if (i == 4 && j == 1) {
                    os << "DuringPreHashes [i=4, j=1]: [" << hash_a[i] << ", " << hash_a[j] << "]" << "\n";
                    os << "DuringPrePair is: " << index << ":" << destination_a[index] << "\n";
                    os << "DuringHamming: " << hamming_distance << "\n";
                }
                if (hamming_distance == 0) {
                    destination_a[index] = 1;
                    //int similarity{ static_cast<int>(1.0 - static_cast<double>(hamming_distance) / (static_cast<double>(hamming_threshold) + 1.0) * 100.0) };
                    
                }
                if (i == 4 && j == 1) {
                    os << "DuringPostHashes [i=4, j=1]: [" << hash_a[i] << ", " << hash_a[j] << "]" << "\n";
                    os << "DuringPostPair is: " << index << ":" << destination_a[index] << "\n";
                }
            }
        };
        cgh.parallel_for<class find_similar_images>(hash_range, kernel);
    });
    queue.wait();
    }
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

    auto files{ GetFiles({ "../vizsla_154/"}) };//, "../maltese_252/", "../vizsla_4048/" }) };
    //auto files{ GetFiles({ "../test/" }) };
    auto hashes{ GetHashes(files) };
    auto pairs{ ProcessDuplicates(hashes, 0) };

    cv::Size window{ 800, 400 };
    auto hash_count{ hashes.size() };

    for (auto i{ 0 }; i < hash_count; ++i) {
        for (auto j{ 0 }; j < i; ++j) {
            auto index = GetLinearIndex(hash_count, i, j);
            if (pairs[index] == 1) {
                auto h = GetHammingDistance(hashes[i], hashes[j]);
                assert(i < files.size());
                assert(j < files.size());
                std::array<cv::Mat, 2> images{
                    Resize(GetImage(files[i]), window.width / 2, window.height),
                    Resize(GetImage(files[j]), window.width / 2, window.height)
                };
                cv::Mat concatenated;
                // Concatenate duplicates images into one big image
                cv::hconcat(images.data(), 2, concatenated);
                cv::imshow("Duplicate Finder", concatenated);
                cv::waitKey(0);
            }
        }
    }

    cv::destroyAllWindows();
        
    

    //for (const auto& pair : pairs) {
    //    std::array<cv::Mat, 2> images{
    //        Resize(GetImage(pair.duplicates.first.path), window.width / 2, window.height),
    //        Resize(GetImage(pair.duplicates.second.path), window.width / 2, window.height)
    //    };
    //    cv::Mat concatenated;
    //    // Concatenate duplicates images into one big image
    //    cv::hconcat(images.data(), 2, concatenated);
    //    cv::imshow("Duplicate Finder", concatenated);
    //    cv::waitKey(0);
    //}

}