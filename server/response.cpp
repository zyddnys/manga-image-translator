#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <cstring>
#include <cassert>

struct Color {
    uint8_t fg[3];
    uint8_t bg[3];
};

struct Translation {
    uint32_t min_x;
    uint32_t min_y;
    uint32_t max_x;
    uint32_t max_y;
    bool is_bulleted_list;
    uint32_t angle;
    float prob;
    Color text_color;
    std::unordered_map<std::string, std::string> text;
    std::vector<uint8_t> background;
};

struct TranslationResponse {
    std::vector<Translation> translations;
};

uint32_t read_u32(const std::vector<uint8_t>& bytes, size_t& offset) {
    uint32_t value;
    std::memcpy(&value, &bytes[offset], sizeof(value));
    value = value;
    offset += 4;
    return value;
}

float read_f32(const std::vector<uint8_t>& bytes, size_t& offset) {
    float value;
    std::memcpy(&value, &bytes[offset], sizeof(value));
    value = *reinterpret_cast<uint32_t*>(&value);
    offset += 4;
    return value;
}

uint8_t read_u8(const std::vector<uint8_t>& bytes, size_t& offset) {
    uint8_t value = bytes[offset];
    offset += 1;
    return value;
}

bool read_bool(const std::vector<uint8_t>& bytes, size_t& offset) {
    bool value = bytes[offset] != 0;
    offset += 1;
    return value;
}

std::vector<uint8_t> read_chunk(const std::vector<uint8_t>& bytes, size_t& offset) {
    uint32_t size = read_u32(bytes, offset);
    std::vector<uint8_t> value(bytes.begin() + offset, bytes.begin() + offset + size);
    offset += size;
    return value;
}

std::string read_str(const std::vector<uint8_t>& bytes, size_t& offset) {
    std::vector<uint8_t> chunk = read_chunk(bytes, offset);
    std::string result(chunk.begin(), chunk.end());
    return result;
}

std::unordered_map<std::string, std::string> read_map(const std::vector<uint8_t>& bytes, size_t& offset) {
    uint32_t count = read_u32(bytes, offset);
    std::unordered_map<std::string, std::string> map;
    for (uint32_t i = 0; i < count; ++i) {
        std::string key = read_str(bytes, offset);
        std::string value = read_str(bytes, offset);
        map[key] = value;
    }
    return map;
}

Translation from_bytes(const std::vector<uint8_t>& bytes, size_t& offset) {
    Translation translation;
    translation.min_x = read_u32(bytes, offset);
    translation.min_y = read_u32(bytes, offset);
    translation.max_x = read_u32(bytes, offset);
    translation.max_y = read_u32(bytes, offset);
    translation.is_bulleted_list = read_bool(bytes, offset);
    translation.angle = read_u32(bytes, offset);
    translation.prob = read_f32(bytes, offset);
    for (int i = 0; i < 3; ++i) {
        translation.text_color.fg[i] = read_u8(bytes, offset);
        translation.text_color.bg[i] = read_u8(bytes, offset);
    }
    translation.text = read_map(bytes, offset);
    translation.background = read_chunk(bytes, offset);
    return translation;
}

TranslationResponse from_bytes_response(const std::vector<uint8_t>& bytes) {
    size_t offset = 0;
    uint32_t count = read_u32(bytes, offset);
    TranslationResponse response;

    for (uint32_t i = 0; i < count; ++i) {
        response.translations.push_back(from_bytes(bytes, offset));
    }
    return response;
}

int main() {
    std::vector<uint8_t> bytes = {/* byte data here */};
    TranslationResponse data = from_bytes_response(bytes);
    return 0;
}
