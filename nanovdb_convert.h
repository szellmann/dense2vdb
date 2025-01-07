// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file   nanovdb_convert.cc

    \author Ken Museth

    \date   May 21, 2020

    \brief  Command-line tool that converts between openvdb and nanovdb files
*/

#include <string>
#include <algorithm>
#include <cctype>

#include <nanovdb/io/IO.h> // this is required to read (and write) NanoVDB files on the host
#define NANOVDB_USE_OPENVDB
#include <nanovdb/tools/CreateNanoGrid.h>
#include <nanovdb/tools/NanoToOpenVDB.h>

static
nanovdb::GridHandle<nanovdb::HostBuffer> nanovdb_convert(openvdb::GridPtrVec *grids)
{
    nanovdb::GridHandle<nanovdb::HostBuffer> handleOut;

    nanovdb::io::Codec       codec = nanovdb::io::Codec::NONE;// compression codec for the file
    nanovdb::tools::StatsMode       sMode = nanovdb::tools::StatsMode::Default;
    nanovdb::CheckMode    cMode = nanovdb::CheckMode::Default;
    nanovdb::GridType        qMode = nanovdb::GridType::Unknown;//specify the quantization mode
    bool                     verbose = false, dither = false, absolute = true;
    float                    tolerance = -1.0f;

    openvdb::initialize();

    // Note, unlike OpenVDB, NanoVDB allows for multiple write operations into the same output file stream.
    // Hence, NanoVDB grids can be read, converted and written to file one at a time whereas all
    // the OpenVDB grids has to be written to file in a single operation.

    auto openToNano = [&](const openvdb::GridBase::Ptr& base)
    {
        using SrcGridT = openvdb::FloatGrid;
        if (auto floatGrid = openvdb::GridBase::grid<SrcGridT>(base)) {
            nanovdb::tools::CreateNanoGrid<SrcGridT> s(*floatGrid);
            s.setStats(sMode);
            s.setChecksum(cMode);
            s.enableDithering(dither);
            s.setVerbose(verbose ? 1 : 0);
            switch (qMode) {
            case nanovdb::GridType::Fp4:
                return s.getHandle<nanovdb::Fp4>();
            case nanovdb::GridType::Fp8:
                return s.getHandle<nanovdb::Fp8>();
            case nanovdb::GridType::Fp16:
                return s.getHandle<nanovdb::Fp16>();
            case nanovdb::GridType::FpN:
                if (absolute) {
                    return s.getHandle<nanovdb::FpN>(nanovdb::tools::AbsDiff(tolerance));
                } else {
                    return s.getHandle<nanovdb::FpN>(nanovdb::tools::RelDiff(tolerance));
                }
            default:
                break;
            }// end of switch
        }
        return nanovdb::tools::openToNanoVDB(base, sMode, cMode, verbose ? 1 : 0);
    };
    try {
        std::vector<nanovdb::GridHandle<nanovdb::HostBuffer> > handles;
        for (auto& grid : *grids) {
            if (verbose) {
                std::cout << "Converting OpenVDB grid named \"" << grid->getName() << "\" to NanoVDB" << std::endl;
            }
            handles.push_back(openToNano(grid));
        } // loop over OpenVDB grids
        handleOut = nanovdb::mergeGrids<nanovdb::HostBuffer, std::vector>(handles);
    }
    catch (const std::exception& e) {
        std::cerr << "An exception occurred: \"" << e.what() << "\"" << std::endl;
    }
    catch (...) {
        std::cerr << "Exception oof unexpected type caught" << std::endl;
    }

    return handleOut;
}
