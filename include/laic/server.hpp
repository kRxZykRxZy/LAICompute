#pragma once
#include <cstdint>
#include <memory>
#include <string>
namespace laic {
class ApiServer {
public:
    explicit ApiServer(uint16_t port=8080, std::string model_dir="models");
    ~ApiServer();
    ApiServer(const ApiServer&)=delete;
    ApiServer& operator=(const ApiServer&)=delete;
    void run();
    void stop() noexcept;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
