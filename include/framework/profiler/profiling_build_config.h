/*************************************************************************************************************************
 * Copyright 2025 Grifcc
 *
 * GryFlux Framework - Profiling Build Config
 *************************************************************************************************************************/
#pragma once

#if !defined(GRYFLUX_BUILD_PROFILING)
#define GRYFLUX_BUILD_PROFILING 0
#endif

namespace GryFlux
{
    namespace Profiling
    {
        // Build-time switch:
        // - Enable profiling: compile with -DGRYFLUX_BUILD_PROFILING=1 and rebuild.
        // - Disable profiling: compile with -DGRYFLUX_BUILD_PROFILING=0 and rebuild.
        // Trade-off: cannot toggle via CLI at runtime.
        inline constexpr bool kBuildProfiling = (GRYFLUX_BUILD_PROFILING != 0);
    } // namespace Profiling
} // namespace GryFlux
