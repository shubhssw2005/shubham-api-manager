from conan import ConanFile
from conan.tools.cmake import CMakeToolchain, CMake, cmake_layout, CMakeDeps


class UltraLowLatencySystemConan(ConanFile):
    name = "ultra-low-latency-system"
    version = "1.0.0"
    package_type = "application"

    # Optional metadata
    license = "MIT"
    author = "Ultra Performance Team"
    url = "https://github.com/company/ultra-low-latency-system"
    description = "Ultra low-latency C++ system for sub-millisecond response times"
    topics = ("performance", "low-latency", "cpp", "dpdk", "cuda")

    # Binary configuration
    settings = "os", "compiler", "build_type", "arch"
    options = {
        "shared": [True, False],
        "fPIC": [True, False],
        "with_cuda": [True, False],
        "with_dpdk": [True, False],
        "with_rdma": [True, False]
    }
    default_options = {
        "shared": False,
        "fPIC": True,
        "with_cuda": True,
        "with_dpdk": True,
        "with_rdma": False
    }

    # Sources are located in the same place as this recipe, copy them to the recipe
    exports_sources = "CMakeLists.txt", "src/*", "include/*", "tests/*", "config/*"

    def requirements(self):
        # Core dependencies
        self.requires("boost/1.82.0")
        self.requires("fmt/10.1.1")
        self.requires("spdlog/1.12.0")
        self.requires("nlohmann_json/3.11.2")
        self.requires("openssl/3.1.3")
        
        # Performance libraries
        self.requires("jemalloc/5.3.0")
        self.requires("mimalloc/2.1.2")
        
        # Networking
        self.requires("libcurl/8.4.0")
        
        # Testing
        self.requires("gtest/1.14.0")
        self.requires("benchmark/1.8.3")
        
        # Monitoring
        self.requires("prometheus-cpp/1.1.0")

    def build_requirements(self):
        self.tool_requires("cmake/3.27.7")

    def layout(self):
        cmake_layout(self)

    def generate(self):
        deps = CMakeDeps(self)
        deps.generate()
        tc = CMakeToolchain(self)
        tc.variables["WITH_CUDA"] = self.options.with_cuda
        tc.variables["WITH_DPDK"] = self.options.with_dpdk
        tc.variables["WITH_RDMA"] = self.options.with_rdma
        tc.generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def package(self):
        cmake = CMake(self)
        cmake.install()