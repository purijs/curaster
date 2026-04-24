#pragma once

#include <cstddef>
#include "../../include/pipeline.h"
#include "../../include/pinned_arena.h"

extern int         g_num_threads;
extern size_t      g_pinned_budget;
extern PinnedArena g_pinned_arena;

void   init_ram_budget();
size_t get_available_ram();



void release_warp_pool();    
void release_halo_pool();    
void release_stack_pool();   
void release_zonal_pool();   

void run_engine_ex(const std::string& input_file, PipelineCtx& ctx, bool verbose);
void run_engine_reproject(const std::string& input_file, PipelineCtx& ctx, bool verbose);
