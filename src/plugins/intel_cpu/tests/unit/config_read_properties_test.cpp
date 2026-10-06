// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>

#include "config.h"
#include "openvino/core/any.hpp"
#include "openvino/core/except.hpp"
#include "openvino/runtime/internal_properties.hpp"
#include "openvino/runtime/properties.hpp"

using namespace ov::intel_cpu;

class ConfigReadPropertiesTests : public ::testing::Test {
protected:
    Config config;
};

TEST_F(ConfigReadPropertiesTests, numStreamsIsStored) {
    config.readProperties({{ov::num_streams.name(), ov::streams::Num(4)}});
    EXPECT_EQ(config.streams, 4);
    EXPECT_TRUE(config.streamsChanged);
}

TEST_F(ConfigReadPropertiesTests, inferenceNumThreadsIsStored) {
    config.readProperties({{ov::inference_num_threads.name(), 2}});
    EXPECT_EQ(config.threads, 2);
}

TEST_F(ConfigReadPropertiesTests, threadsPerStreamIsStored) {
    config.readProperties({{ov::internal::threads_per_stream.name(), size_t{3}}});
    EXPECT_EQ(config.threadsPerStream, 3);
}

TEST_F(ConfigReadPropertiesTests, allStreamExecutorKeysTogether) {
    config.readProperties({{ov::num_streams.name(), ov::streams::Num(2)},
                           {ov::inference_num_threads.name(), 8},
                           {ov::internal::threads_per_stream.name(), size_t{4}}});
    EXPECT_EQ(config.streams, 2);
    EXPECT_EQ(config.threads, 8);
    EXPECT_EQ(config.threadsPerStream, 4);
}

TEST_F(ConfigReadPropertiesTests, numStreamsAutoSwitchesToThroughputHint) {
    config.readProperties({{ov::num_streams.name(), ov::streams::AUTO}});
    EXPECT_EQ(config.hintPerfMode, ov::hint::PerformanceMode::THROUGHPUT);
    EXPECT_TRUE(config.changedHintPerfMode);
    EXPECT_FALSE(config.streamsChanged);
}

TEST_F(ConfigReadPropertiesTests, performanceModeIsHandled) {
    config.readProperties({{ov::hint::performance_mode.name(), ov::hint::PerformanceMode::THROUGHPUT}});
    EXPECT_EQ(config.hintPerfMode, ov::hint::PerformanceMode::THROUGHPUT);
}

TEST_F(ConfigReadPropertiesTests, enableProfilingIsHandled) {
    config.readProperties({{ov::enable_profiling.name(), true}});
    EXPECT_TRUE(config.collectPerfCounters);
}

TEST_F(ConfigReadPropertiesTests, numRequestsIsHandled) {
    config.readProperties({{ov::hint::num_requests.name(), 5}});
    EXPECT_EQ(config.hintNumRequests, 5U);
}

TEST_F(ConfigReadPropertiesTests, unknownKeyThrows) {
    try {
        config.readProperties({{"SOME_UNKNOWN_PROPERTY", 1}});
        FAIL() << "Expected ov::Exception";
    } catch (const ov::Exception& ex) {
        EXPECT_NE(std::string(ex.what()).find("Unsupported property"), std::string::npos) << ex.what();
    }
}

TEST_F(ConfigReadPropertiesTests, negativeInferenceNumThreadsThrows) {
    EXPECT_THROW(config.readProperties({{ov::inference_num_threads.name(), -1}}), ov::Exception);
}

TEST_F(ConfigReadPropertiesTests, invalidPerformanceModeThrows) {
    EXPECT_THROW(config.readProperties({{ov::hint::performance_mode.name(), std::string("INVALID_MODE")}}),
                 ov::Exception);
}
