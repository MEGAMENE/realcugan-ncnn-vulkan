// realcugan implemented with ncnn library

#include "realcugan.h"

#include <algorithm>
#include <vector>
#include <unordered_map>
#include <cstring>

// ncnn
#include "cpu.h"
#include "pipelinecache.h"

#include "realcugan_preproc.comp.hex.h"
#include "realcugan_postproc.comp.hex.h"
#include "realcugan_4x_postproc.comp.hex.h"
#include "realcugan_preproc_tta.comp.hex.h"
#include "realcugan_postproc_tta.comp.hex.h"
#include "realcugan_4x_postproc_tta.comp.hex.h"

class FeatureCache
{
public:
    void clear()
    {
        gpu_cache.clear();
        cpu_cache.clear();
    }

    static inline uint64_t make_key(int yi, int xi, int ti, const std::string& name)
    {
        uint32_t name_id = 0;
        if (name.size() >= 4 && name[0] == 'g' && name[1] == 'a' && name[2] == 'p')
        {
            name_id = (uint32_t)(name[3] - '0');
        }
        else
        {
            for (char c : name)
                name_id = (name_id * 31) + (uint8_t)c;
        }
        return ((uint64_t)(uint16_t)yi << 48) | ((uint64_t)(uint16_t)xi << 32) | ((uint64_t)(uint8_t)ti << 24) | ((uint64_t)name_id & 0xffffffULL);
    }

    void load(int yi, int xi, int ti, const std::string& name, ncnn::VkMat& feat)
    {
        feat = gpu_cache[make_key(yi, xi, ti, name)];
    }

    void save(int yi, int xi, int ti, const std::string& name, ncnn::VkMat& feat)
    {
        gpu_cache[make_key(yi, xi, ti, name)] = feat;
    }

    void load(int yi, int xi, int ti, const std::string& name, ncnn::Mat& feat)
    {
        feat = cpu_cache[make_key(yi, xi, ti, name)];
    }

    void save(int yi, int xi, int ti, const std::string& name, ncnn::Mat& feat)
    {
        cpu_cache[make_key(yi, xi, ti, name)] = feat;
    }

public:
    std::unordered_map<uint64_t, ncnn::VkMat> gpu_cache;
    std::unordered_map<uint64_t, ncnn::Mat> cpu_cache;
};

#if _WIN32
#include <direct.h>
static std::wstring get_pipeline_cache_path(int gpuid)
{
    std::wstring fname = L"\\pipeline_gpu" + std::to_wstring(gpuid) + L".bin";
    const wchar_t* localappdata = _wgetenv(L"LOCALAPPDATA");
    if (localappdata)
    {
        std::wstring dir = std::wstring(localappdata) + L"\\realcugan-ncnn-vulkan";
        _wmkdir(dir.c_str());
        return dir + fname;
    }
    return L"pipeline_gpu" + std::to_wstring(gpuid) + L".bin";
}
#else
#include <sys/stat.h>
static std::string get_pipeline_cache_path(int gpuid)
{
    std::string fname = "/pipeline_gpu" + std::to_string(gpuid) + ".bin";
    const char* xdg_cache = getenv("XDG_CACHE_HOME");
    if (xdg_cache)
    {
        std::string dir = std::string(xdg_cache) + "/realcugan-ncnn-vulkan";
        mkdir(dir.c_str(), 0755);
        return dir + fname;
    }
    const char* home = getenv("HOME");
    if (home)
    {
        std::string dir = std::string(home) + "/.cache";
        mkdir(dir.c_str(), 0755);
        dir += "/realcugan-ncnn-vulkan";
        mkdir(dir.c_str(), 0755);
        return dir + fname;
    }
    return "pipeline_gpu" + std::to_string(gpuid) + ".bin";
}
#endif

template <typename StringType>
static bool check_pro_model(const StringType& path)
{
    size_t last_sep = path.find_last_of(
#if _WIN32
        L"/\\"
#else
        "/\\"
#endif
    );
    StringType filename = (last_sep == StringType::npos) ? path : path.substr(last_sep + 1);
    StringType parent_dir;
    if (last_sep != StringType::npos)
    {
        size_t prev_sep = path.find_last_of(
#if _WIN32
            L"/\\",
#else
            "/\\",
#endif
            last_sep == 0 ? StringType::npos : last_sep - 1
        );
        if (prev_sep == StringType::npos)
            parent_dir = path.substr(0, last_sep);
        else
            parent_dir = path.substr(prev_sep + 1, last_sep - prev_sep - 1);
    }

#if _WIN32
    const wchar_t* token = L"-pro";
#else
    const char* token = "-pro";
#endif
    if (parent_dir.find(token) != StringType::npos || filename.find(token) != StringType::npos)
    {
        return true;
    }
    return false;
}

RealCUGAN::RealCUGAN(int _gpuid, int _tta_level, int num_threads)
{
    vkdev = _gpuid == -1 ? 0 : ncnn::get_gpu_device(_gpuid);
    gpuid = _gpuid;
    if (vkdev)
    {
        const char* dev_name = vkdev->info.device_name();
        if (vkdev->info.type() == 3 || (dev_name && (strstr(dev_name, "llvmpipe") || strstr(dev_name, "lavapipe"))))
        {
            vkdev = 0;
        }
    }

    net.opt.num_threads = num_threads;

    realcugan_preproc = 0;
    realcugan_postproc = 0;
    realcugan_4x_postproc = 0;
    bicubic_2x = 0;
    bicubic_3x = 0;
    bicubic_4x = 0;
    tta_level = _tta_level;
    use_fp16_arithmetic = false;
#if _WIN32
    bgr_mode = true;
#else
    bgr_mode = false;
#endif
    is_pro_model = false;
    noise = -1;
    scale = 2;
    tilesize = 0;
    prepadding = 0;
    syncgap = 3;
}

RealCUGAN::~RealCUGAN()
{
    if (vkdev)
    {
        auto cache_path = get_pipeline_cache_path(gpuid);
        int ret = vkdev->get_pipeline_cache()->save_cache(cache_path.c_str());
#if _WIN32
        std::wstring fallback = L"pipeline_gpu" + std::to_wstring(gpuid) + L".bin";
        if (ret != 0 && cache_path != fallback)
        {
            vkdev->get_pipeline_cache()->save_cache(fallback.c_str());
        }
#else
        std::string fallback = "pipeline_gpu" + std::to_string(gpuid) + ".bin";
        if (ret != 0 && cache_path != fallback)
        {
            vkdev->get_pipeline_cache()->save_cache(fallback.c_str());
        }
#endif
    }

    // cleanup preprocess and postprocess pipeline
    {
        delete realcugan_preproc;
        delete realcugan_postproc;
        delete realcugan_4x_postproc;
    }

    if (bicubic_2x)
    {
        if (vkdev)
            bicubic_2x->destroy_pipeline(net.opt);
        delete bicubic_2x;
    }

    if (bicubic_3x)
    {
        if (vkdev)
            bicubic_3x->destroy_pipeline(net.opt);
        delete bicubic_3x;
    }

    if (bicubic_4x)
    {
        if (vkdev)
            bicubic_4x->destroy_pipeline(net.opt);
        delete bicubic_4x;
    }
}

#if _WIN32
int RealCUGAN::load(const std::wstring& parampath, const std::wstring& modelpath)
#else
int RealCUGAN::load(const std::string& parampath, const std::string& modelpath)
#endif
{
    is_pro_model = false;
    if (this->scale == 2 || this->scale == 3)
    {
        if (check_pro_model(parampath) || check_pro_model(modelpath))
        {
            is_pro_model = true;
        }
    }

    net.opt.use_vulkan_compute = vkdev ? true : false;
    net.opt.use_fp16_packed = vkdev ? vkdev->info.support_fp16_packed() : true;
    net.opt.use_fp16_storage = vkdev ? vkdev->info.support_fp16_storage() : false;
    net.opt.use_fp16_arithmetic = vkdev ? (use_fp16_arithmetic && vkdev->info.support_fp16_arithmetic()) : false;
    net.opt.use_int8_storage = vkdev ? vkdev->info.support_int8_storage() : false;
    net.opt.use_cooperative_matrix = vkdev ? vkdev->info.support_cooperative_matrix() : false;
    net.opt.use_shader_local_memory = true;
    net.opt.flush_denormals = 3;
    net.opt.use_winograd_convolution = true;
    net.opt.use_sgemm_convolution = true;

    if (vkdev)
        net.set_vulkan_device(vkdev);

    auto cache_path = get_pipeline_cache_path(gpuid);
    if (vkdev)
    {
        int ret = vkdev->get_pipeline_cache()->load_cache(cache_path.c_str());
#if _WIN32
        std::wstring fallback = L"pipeline_gpu" + std::to_wstring(gpuid) + L".bin";
        if (ret != 0 && cache_path != fallback)
        {
            vkdev->get_pipeline_cache()->load_cache(fallback.c_str());
        }
#else
        std::string fallback = "pipeline_gpu" + std::to_string(gpuid) + ".bin";
        if (ret != 0 && cache_path != fallback)
        {
            vkdev->get_pipeline_cache()->load_cache(fallback.c_str());
        }
#endif
    }

#if _WIN32
    {
        FILE* fp = _wfopen(parampath.c_str(), L"rb");
        if (!fp)
        {
            fwprintf(stderr, L"_wfopen %ls failed\n", parampath.c_str());
            return -1;
        }

        int ret = net.load_param(fp);
        fclose(fp);
        if (ret != 0)
        {
            fwprintf(stderr, L"load_param %ls failed\n", parampath.c_str());
            return -1;
        }
    }
    {
        FILE* fp = _wfopen(modelpath.c_str(), L"rb");
        if (!fp)
        {
            fwprintf(stderr, L"_wfopen %ls failed\n", modelpath.c_str());
            return -1;
        }

        int ret = net.load_model(fp);
        fclose(fp);
        if (ret != 0)
        {
            fwprintf(stderr, L"load_model %ls failed\n", modelpath.c_str());
            return -1;
        }
    }
#else
    if (net.load_param(parampath.c_str()) != 0)
    {
        fprintf(stderr, "load_param %s failed\n", parampath.c_str());
        return -1;
    }
    if (net.load_model(modelpath.c_str()) != 0)
    {
        fprintf(stderr, "load_model %s failed\n", modelpath.c_str());
        return -1;
    }
#endif

    // initialize preprocess and postprocess pipeline
    if (vkdev)
    {
        std::vector<ncnn::vk_specialization_type> specializations((tta_level > 1) ? 3 : 2);
        specializations[0].i = bgr_mode ? 1 : 0;
        specializations[1].i = is_pro_model ? 1 : 0;
        if (tta_level > 1)
        {
            specializations[2].i = tta_level;
        }

        std::vector<ncnn::vk_specialization_type> specializations_4x((tta_level > 1) ? 3 : 2);
        specializations_4x[0].i = bgr_mode ? 1 : 0;
        specializations_4x[1].i = is_pro_model ? 1 : 0;
        if (tta_level > 1)
        {
            specializations_4x[2].i = tta_level;
        }

        const int tta_idx = (tta_level > 1) ? 1 : 0;

        {
            if (spirv_preproc[tta_idx].empty())
            {
                if (tta_level > 1)
                    compile_spirv_module(realcugan_preproc_tta_comp_data, sizeof(realcugan_preproc_tta_comp_data), net.opt, spirv_preproc[tta_idx]);
                else
                    compile_spirv_module(realcugan_preproc_comp_data, sizeof(realcugan_preproc_comp_data), net.opt, spirv_preproc[tta_idx]);
            }

            realcugan_preproc = new ncnn::Pipeline(vkdev);
            realcugan_preproc->set_optimal_local_size_xyz(8, 8, 3);
            realcugan_preproc->create(spirv_preproc[tta_idx].data(), spirv_preproc[tta_idx].size() * 4, specializations);
        }

        {
            if (spirv_postproc[tta_idx].empty())
            {
                if (tta_level > 1)
                    compile_spirv_module(realcugan_postproc_tta_comp_data, sizeof(realcugan_postproc_tta_comp_data), net.opt, spirv_postproc[tta_idx]);
                else
                    compile_spirv_module(realcugan_postproc_comp_data, sizeof(realcugan_postproc_comp_data), net.opt, spirv_postproc[tta_idx]);
            }

            realcugan_postproc = new ncnn::Pipeline(vkdev);
            realcugan_postproc->set_optimal_local_size_xyz(8, 8, 3);
            realcugan_postproc->create(spirv_postproc[tta_idx].data(), spirv_postproc[tta_idx].size() * 4, specializations);
        }

        {
            if (spirv_4x_postproc[tta_idx].empty())
            {
                if (tta_level > 1)
                    compile_spirv_module(realcugan_4x_postproc_tta_comp_data, sizeof(realcugan_4x_postproc_tta_comp_data), net.opt, spirv_4x_postproc[tta_idx]);
                else
                    compile_spirv_module(realcugan_4x_postproc_comp_data, sizeof(realcugan_4x_postproc_comp_data), net.opt, spirv_4x_postproc[tta_idx]);
            }

            realcugan_4x_postproc = new ncnn::Pipeline(vkdev);
            realcugan_4x_postproc->set_optimal_local_size_xyz(8, 8, 3);
            realcugan_4x_postproc->create(spirv_4x_postproc[tta_idx].data(), spirv_4x_postproc[tta_idx].size() * 4, specializations_4x);
        }
    }

    // bicubic 2x/3x/4x for alpha channel
    {
        bicubic_2x = ncnn::create_layer("Interp");
        bicubic_2x->vkdev = vkdev;

        ncnn::ParamDict pd;
        pd.set(0, 3);// bicubic
        pd.set(1, 2.f);
        pd.set(2, 2.f);
        bicubic_2x->load_param(pd);

        if (vkdev)
            bicubic_2x->create_pipeline(net.opt);
    }
    {
        bicubic_3x = ncnn::create_layer("Interp");
        bicubic_3x->vkdev = vkdev;

        ncnn::ParamDict pd;
        pd.set(0, 3);// bicubic
        pd.set(1, 3.f);
        pd.set(2, 3.f);
        bicubic_3x->load_param(pd);

        if (vkdev)
            bicubic_3x->create_pipeline(net.opt);
    }
    {
        bicubic_4x = ncnn::create_layer("Interp");
        bicubic_4x->vkdev = vkdev;

        ncnn::ParamDict pd;
        pd.set(0, 3);// bicubic
        pd.set(1, 4.f);
        pd.set(2, 4.f);
        bicubic_4x->load_param(pd);

        if (vkdev)
            bicubic_4x->create_pipeline(net.opt);
    }

    return 0;
}

int RealCUGAN::process(const ncnn::Mat& inimage, ncnn::Mat& outimage) const
{
    bool single_tile_failed = false;
    int current_tilesize = this->tilesize;

    while (current_tilesize >= 32)
    {
        const int w = inimage.w;
        const int h = inimage.h;
        const int channels = inimage.elempack;

        int64_t padded_area = (int64_t)(w + 2 * prepadding) * (h + 2 * prepadding);
        double bpp = (scale == 3) ? 18432.0 : ((scale == 4) ? 9011.0 : 7004.0);

        bool can_single_tile = false;
        if (!single_tile_failed)
        {
            int64_t max_tile_area = (int64_t)(current_tilesize + 2 * prepadding) * (current_tilesize + 2 * prepadding);
            if (vkdev)
            {
                uint32_t heap_budget_mb = vkdev->get_heap_budget();
                double safe_vram = (double)heap_budget_mb * 1024.0 * 1024.0 * 0.70;
                bool is_igpu = (vkdev->info.type() == 1);
                int max_dim = (scale == 3) ? 256 : 384;
                bool dim_ok = !is_igpu || (w <= max_dim && h <= max_dim);
                if (current_tilesize > 0)
                {
                    if (padded_area <= max_tile_area && dim_ok)
                    {
                        can_single_tile = true;
                    }
                }
                else
                {
                    if ((double)padded_area * bpp <= safe_vram && dim_ok)
                    {
                        can_single_tile = true;
                    }
                }
            }
            else
            {
                if (current_tilesize > 0 && padded_area <= max_tile_area)
                {
                    can_single_tile = true;
                }
            }
        }

        bool syncgap_needed = !can_single_tile && (current_tilesize < std::max(w, h));

        if (!vkdev)
        {
            // cpu only
            if (syncgap_needed && syncgap)
            {
                int effective_syncgap = syncgap;
                int xt = (w + current_tilesize - 1) / current_tilesize;
                int yt = (h + current_tilesize - 1) / current_tilesize;
                if (effective_syncgap == 3 && (xt < 3 || yt < 3))
                    effective_syncgap = 2;

                if (effective_syncgap == 1)
                    return process_cpu_se(inimage, outimage, current_tilesize);
                if (effective_syncgap == 2)
                    return process_cpu_se_rough(inimage, outimage, current_tilesize);
                if (effective_syncgap == 3)
                    return process_cpu_se_very_rough(inimage, outimage, current_tilesize);
            }
            else
                return process_cpu(inimage, outimage);
        }

        if (noise == -1 && scale == 1)
        {
            if (outimage.data && inimage.data && outimage.data != inimage.data)
            {
                memcpy(outimage.data, inimage.data, (size_t)inimage.w * inimage.h * inimage.elempack);
            }
            else
            {
                outimage = inimage;
            }
            return 0;
        }

        int xtiles = can_single_tile ? 1 : ((w + current_tilesize - 1) / current_tilesize);
        int ytiles = can_single_tile ? 1 : ((h + current_tilesize - 1) / current_tilesize);

        if (syncgap_needed && syncgap)
        {
            int effective_syncgap = syncgap;
            if (effective_syncgap == 3 && (xtiles < 3 || ytiles < 3))
            {
                effective_syncgap = 2;
            }

            int ret = 0;
            if (effective_syncgap == 1)
                ret = process_se(inimage, outimage, current_tilesize);
            else if (effective_syncgap == 2)
                ret = process_se_rough(inimage, outimage, current_tilesize);
            else if (effective_syncgap == 3)
                ret = process_se_very_rough(inimage, outimage, current_tilesize);

            if (ret != 0)
            {
                if (current_tilesize <= 32)
                {
                    fprintf(stderr, "process failed even at tilesize 32 (error code %d)\n", ret);
                    return ret;
                }
                int next_tilesize = std::max(32, (current_tilesize / 2 / 32) * 32);
                if (next_tilesize == current_tilesize) next_tilesize = 32;
                fprintf(stderr, "warning: extract failed (error code %d, possible VRAM OOM), halving tilesize from %d to %d and retrying\n", ret, current_tilesize, next_tilesize);
                current_tilesize = next_tilesize;
                continue;
            }
            return 0;
        }

        const unsigned char* pixeldata = (const unsigned char*)inimage.data;
        const int tile_w = can_single_tile ? w : (((w + xtiles - 1) / xtiles + 3) & ~3);
        const int tile_h = can_single_tile ? h : (((h + ytiles - 1) / ytiles + 3) & ~3);
        xtiles = can_single_tile ? 1 : ((w + tile_w - 1) / tile_w);
        ytiles = can_single_tile ? 1 : ((h + tile_h - 1) / tile_h);

        ncnn::VkAllocator* blob_vkallocator = vkdev->acquire_blob_allocator();
        ncnn::VkAllocator* staging_vkallocator = vkdev->acquire_staging_allocator();

        ncnn::Option opt = net.opt;
        opt.blob_vkallocator = blob_vkallocator;
        opt.workspace_vkallocator = blob_vkallocator;
        opt.staging_vkallocator = staging_vkallocator;

        const size_t in_out_tile_elemsize = (opt.use_fp16_storage || opt.use_fp16_packed) ? 2u : 4u;
        bool extract_failed = false;
        int extract_ret = 0;

        for (int yi = 0; yi < ytiles; yi++)
        {
            const int tile_h_nopad = std::min((yi + 1) * tile_h, h) - yi * tile_h;

            int prepadding_bottom = prepadding;
            if (scale == 1 || scale == 3)
            {
                prepadding_bottom += (tile_h_nopad + 3) / 4 * 4 - tile_h_nopad;
            }
            if (scale == 2 || scale == 4)
            {
                prepadding_bottom += (tile_h_nopad + 1) / 2 * 2 - tile_h_nopad;
            }

            int in_tile_y0 = std::max(yi * tile_h - prepadding, 0);
            int in_tile_y1 = std::min((yi + 1) * tile_h + prepadding_bottom, h);

            ncnn::Mat in;
            if ((opt.use_fp16_storage || opt.use_fp16_packed) && opt.use_int8_storage)
            {
                in = ncnn::Mat(w, (in_tile_y1 - in_tile_y0), (unsigned char*)pixeldata + (size_t)in_tile_y0 * w * channels, (size_t)channels, 1);
            }
            else
            {
                if (channels == 3)
                {
                    in = ncnn::Mat::from_pixels(pixeldata + (size_t)in_tile_y0 * w * channels, bgr_mode ? ncnn::Mat::PIXEL_BGR2RGB : ncnn::Mat::PIXEL_RGB, w, (in_tile_y1 - in_tile_y0));
                }
                if (channels == 4)
                {
                    in = ncnn::Mat::from_pixels(pixeldata + (size_t)in_tile_y0 * w * channels, bgr_mode ? ncnn::Mat::PIXEL_BGRA2RGBA : ncnn::Mat::PIXEL_RGBA, w, (in_tile_y1 - in_tile_y0));
                }
            }

            ncnn::VkCompute cmd(vkdev);

            // upload
            ncnn::VkMat in_gpu;
            {
                cmd.record_clone(in, in_gpu, opt);

                if (xtiles > 1)
                {
                    int ret = cmd.submit_and_wait();
                    if (ret != 0)
                    {
                        extract_failed = true;
                        extract_ret = ret;
                        break;
                    }
                    cmd.reset();
                }
            }

            if (extract_failed) break;

            int out_tile_y0 = yi * tile_h;
            int out_tile_y1 = std::min((yi + 1) * tile_h, h);

            ncnn::VkMat out_gpu;
            if ((opt.use_fp16_storage || opt.use_fp16_packed) && opt.use_int8_storage)
            {
                out_gpu.create(w * scale, (out_tile_y1 - out_tile_y0) * scale, (size_t)channels, 1, blob_vkallocator);
            }
            else
            {
                out_gpu.create(w * scale, (out_tile_y1 - out_tile_y0) * scale, channels, (size_t)4u, 1, blob_vkallocator);
            }

            for (int xi = 0; xi < xtiles; xi++)
            {
                const int tile_w_nopad = std::min((xi + 1) * tile_w, w) - xi * tile_w;

                int prepadding_right = prepadding;
                if (scale == 1 || scale == 3)
                {
                    prepadding_right += (tile_w_nopad + 3) / 4 * 4 - tile_w_nopad;
                }
                if (scale == 2 || scale == 4)
                {
                    prepadding_right += (tile_w_nopad + 1) / 2 * 2 - tile_w_nopad;
                }

                if (tta_level > 1)
                {
                    // preproc
                    ncnn::VkMat in_tile_gpu[8];
                    ncnn::VkMat in_alpha_tile_gpu;
                    ncnn::VkMat dummy_tile_gpu;
                    if (tta_level < 8)
                    {
                        dummy_tile_gpu.create(1, 1, 3, in_out_tile_elemsize, 1, blob_vkallocator);
                    }
                    {
                        // crop tile
                        int tile_x0 = xi * tile_w - prepadding;
                        int tile_x1 = std::min((xi + 1) * tile_w, w) + prepadding_right;
                        int tile_y0 = yi * tile_h - prepadding;
                        int tile_y1 = std::min((yi + 1) * tile_h, h) + prepadding_bottom;

                        in_tile_gpu[0].create(tile_x1 - tile_x0, tile_y1 - tile_y0, 3, in_out_tile_elemsize, 1, blob_vkallocator);
                        in_tile_gpu[1].create(tile_x1 - tile_x0, tile_y1 - tile_y0, 3, in_out_tile_elemsize, 1, blob_vkallocator);
                        if (tta_level >= 4)
                        {
                            in_tile_gpu[2].create(tile_x1 - tile_x0, tile_y1 - tile_y0, 3, in_out_tile_elemsize, 1, blob_vkallocator);
                            in_tile_gpu[3].create(tile_x1 - tile_x0, tile_y1 - tile_y0, 3, in_out_tile_elemsize, 1, blob_vkallocator);
                        }
                        if (tta_level >= 8)
                        {
                            in_tile_gpu[4].create(tile_y1 - tile_y0, tile_x1 - tile_x0, 3, in_out_tile_elemsize, 1, blob_vkallocator);
                            in_tile_gpu[5].create(tile_y1 - tile_y0, tile_x1 - tile_x0, 3, in_out_tile_elemsize, 1, blob_vkallocator);
                            in_tile_gpu[6].create(tile_y1 - tile_y0, tile_x1 - tile_x0, 3, in_out_tile_elemsize, 1, blob_vkallocator);
                            in_tile_gpu[7].create(tile_y1 - tile_y0, tile_x1 - tile_x0, 3, in_out_tile_elemsize, 1, blob_vkallocator);
                        }

                        if (channels == 4)
                        {
                            in_alpha_tile_gpu.create(tile_w_nopad, tile_h_nopad, 1, in_out_tile_elemsize, 1, blob_vkallocator);
                        }

                        std::vector<ncnn::VkMat> bindings(10);
                        bindings[0] = in_gpu;
                        for (int ti = 0; ti < tta_level; ti++)
                        {
                            bindings[1 + ti] = in_tile_gpu[ti];
                        }
                        for (int ti = tta_level; ti < 8; ti++)
                        {
                            bindings[1 + ti] = dummy_tile_gpu;
                        }
                        bindings[9] = in_alpha_tile_gpu;

                        std::vector<ncnn::vk_constant_type> constants(13);
                        constants[0].i = in_gpu.w;
                        constants[1].i = in_gpu.h;
                        constants[2].i = in_gpu.cstep;
                        constants[3].i = in_tile_gpu[0].w;
                        constants[4].i = in_tile_gpu[0].h;
                        constants[5].i = in_tile_gpu[0].cstep;
                        constants[6].i = prepadding;
                        constants[7].i = prepadding;
                        constants[8].i = xi * tile_w;
                        constants[9].i = std::min(yi * tile_h, prepadding);
                        constants[10].i = channels;
                        constants[11].i = in_alpha_tile_gpu.w;
                        constants[12].i = in_alpha_tile_gpu.h;

                        ncnn::VkMat dispatcher;
                        dispatcher.w = in_tile_gpu[0].w;
                        dispatcher.h = in_tile_gpu[0].h;
                        dispatcher.c = channels;

                        cmd.record_pipeline(realcugan_preproc, bindings, constants, dispatcher);
                    }

                    // realcugan
                    ncnn::VkMat out_tile_gpu[8];
                    for (int ti = 0; ti < tta_level; ti++)
                    {
                        ncnn::Extractor ex = net.create_extractor();

                        ex.set_blob_vkallocator(blob_vkallocator);
                        ex.set_workspace_vkallocator(blob_vkallocator);
                        ex.set_staging_vkallocator(staging_vkallocator);

                        ex.input("in0", in_tile_gpu[ti]);

                        int ret = ex.extract("out0", out_tile_gpu[ti], cmd);
                        if (ret != 0)
                        {
                            extract_failed = true;
                            extract_ret = ret;
                            break;
                        }
                    }
                    if (extract_failed) break;

                    ncnn::VkMat out_alpha_tile_gpu;
                    if (channels == 4)
                    {
                        if (scale == 1)
                        {
                            out_alpha_tile_gpu = in_alpha_tile_gpu;
                        }
                        if (scale == 2)
                        {
                            bicubic_2x->forward(in_alpha_tile_gpu, out_alpha_tile_gpu, cmd, opt);
                        }
                        if (scale == 3)
                        {
                            bicubic_3x->forward(in_alpha_tile_gpu, out_alpha_tile_gpu, cmd, opt);
                        }
                        if (scale == 4)
                        {
                            bicubic_4x->forward(in_alpha_tile_gpu, out_alpha_tile_gpu, cmd, opt);
                        }
                    }

                    // postproc
                    if (scale == 4)
                    {
                        std::vector<ncnn::VkMat> bindings(11);
                        bindings[0] = in_gpu;
                        for (int ti = 0; ti < tta_level; ti++)
                        {
                            bindings[1 + ti] = out_tile_gpu[ti];
                        }
                        for (int ti = tta_level; ti < 8; ti++)
                        {
                            bindings[1 + ti] = dummy_tile_gpu;
                        }
                        bindings[9] = out_alpha_tile_gpu;
                        bindings[10] = out_gpu;

                        std::vector<ncnn::vk_constant_type> constants(16);
                        constants[0].i = in_gpu.w;
                        constants[1].i = in_gpu.h;
                        constants[2].i = in_gpu.cstep;
                        constants[3].i = out_tile_gpu[0].w;
                        constants[4].i = out_tile_gpu[0].h;
                        constants[5].i = out_tile_gpu[0].cstep;
                        constants[6].i = out_gpu.w;
                        constants[7].i = out_gpu.h;
                        constants[8].i = out_gpu.cstep;
                        constants[9].i = xi * tile_w;
                        constants[10].i = std::min(yi * tile_h, prepadding);
                        constants[11].i = xi * tile_w * scale;
                        constants[12].i = std::min(tile_w * scale, out_gpu.w - xi * tile_w * scale);
                        constants[13].i = channels;
                        constants[14].i = out_alpha_tile_gpu.w;
                        constants[15].i = out_alpha_tile_gpu.h;

                        ncnn::VkMat dispatcher;
                        dispatcher.w = std::min(tile_w * scale, out_gpu.w - xi * tile_w * scale);
                        dispatcher.h = out_gpu.h;
                        dispatcher.c = channels;

                        cmd.record_pipeline(realcugan_4x_postproc, bindings, constants, dispatcher);
                    }
                    else
                    {
                        std::vector<ncnn::VkMat> bindings(10);
                        for (int ti = 0; ti < tta_level; ti++)
                        {
                            bindings[ti] = out_tile_gpu[ti];
                        }
                        for (int ti = tta_level; ti < 8; ti++)
                        {
                            bindings[ti] = dummy_tile_gpu;
                        }
                        bindings[8] = out_alpha_tile_gpu;
                        bindings[9] = out_gpu;

                        std::vector<ncnn::vk_constant_type> constants(11);
                        constants[0].i = out_tile_gpu[0].w;
                        constants[1].i = out_tile_gpu[0].h;
                        constants[2].i = out_tile_gpu[0].cstep;
                        constants[3].i = out_gpu.w;
                        constants[4].i = out_gpu.h;
                        constants[5].i = out_gpu.cstep;
                        constants[6].i = xi * tile_w * scale;
                        constants[7].i = std::min(tile_w * scale, out_gpu.w - xi * tile_w * scale);
                        constants[8].i = channels;
                        constants[9].i = out_alpha_tile_gpu.w;
                        constants[10].i = out_alpha_tile_gpu.h;

                        ncnn::VkMat dispatcher;
                        dispatcher.w = std::min(tile_w * scale, out_gpu.w - xi * tile_w * scale);
                        dispatcher.h = out_gpu.h;
                        dispatcher.c = channels;

                        cmd.record_pipeline(realcugan_postproc, bindings, constants, dispatcher);
                    }
                }
                else
                {
                    // preproc
                    ncnn::VkMat in_tile_gpu;
                    ncnn::VkMat in_alpha_tile_gpu;
                    {
                        // crop tile
                        int tile_x0 = xi * tile_w - prepadding;
                        int tile_x1 = std::min((xi + 1) * tile_w, w) + prepadding_right;
                        int tile_y0 = yi * tile_h - prepadding;
                        int tile_y1 = std::min((yi + 1) * tile_h, h) + prepadding_bottom;

                        in_tile_gpu.create(tile_x1 - tile_x0, tile_y1 - tile_y0, 3, in_out_tile_elemsize, 1, blob_vkallocator);

                        if (channels == 4)
                        {
                            in_alpha_tile_gpu.create(tile_w_nopad, tile_h_nopad, 1, in_out_tile_elemsize, 1, blob_vkallocator);
                        }

                        std::vector<ncnn::VkMat> bindings(3);
                        bindings[0] = in_gpu;
                        bindings[1] = in_tile_gpu;
                        bindings[2] = in_alpha_tile_gpu;

                        std::vector<ncnn::vk_constant_type> constants(13);
                        constants[0].i = in_gpu.w;
                        constants[1].i = in_gpu.h;
                        constants[2].i = in_gpu.cstep;
                        constants[3].i = in_tile_gpu.w;
                        constants[4].i = in_tile_gpu.h;
                        constants[5].i = in_tile_gpu.cstep;
                        constants[6].i = prepadding;
                        constants[7].i = prepadding;
                        constants[8].i = xi * tile_w;
                        constants[9].i = std::min(yi * tile_h, prepadding);
                        constants[10].i = channels;
                        constants[11].i = in_alpha_tile_gpu.w;
                        constants[12].i = in_alpha_tile_gpu.h;

                        ncnn::VkMat dispatcher;
                        dispatcher.w = in_tile_gpu.w;
                        dispatcher.h = in_tile_gpu.h;
                        dispatcher.c = channels;

                        cmd.record_pipeline(realcugan_preproc, bindings, constants, dispatcher);
                    }

                    // realcugan
                    ncnn::VkMat out_tile_gpu;
                    {
                        ncnn::Extractor ex = net.create_extractor();

                        ex.set_blob_vkallocator(blob_vkallocator);
                        ex.set_workspace_vkallocator(blob_vkallocator);
                        ex.set_staging_vkallocator(staging_vkallocator);

                        ex.input("in0", in_tile_gpu);

                        int ret = ex.extract("out0", out_tile_gpu, cmd);
                        if (ret != 0)
                        {
                            extract_failed = true;
                            extract_ret = ret;
                            break;
                        }
                    }

                    ncnn::VkMat out_alpha_tile_gpu;
                    if (channels == 4)
                    {
                        if (scale == 1)
                        {
                            out_alpha_tile_gpu = in_alpha_tile_gpu;
                        }
                        if (scale == 2)
                        {
                            bicubic_2x->forward(in_alpha_tile_gpu, out_alpha_tile_gpu, cmd, opt);
                        }
                        if (scale == 3)
                        {
                            bicubic_3x->forward(in_alpha_tile_gpu, out_alpha_tile_gpu, cmd, opt);
                        }
                        if (scale == 4)
                        {
                            bicubic_4x->forward(in_alpha_tile_gpu, out_alpha_tile_gpu, cmd, opt);
                        }
                    }

                    // postproc
                    if (scale == 4)
                    {
                        std::vector<ncnn::VkMat> bindings(4);
                        bindings[0] = in_gpu;
                        bindings[1] = out_tile_gpu;
                        bindings[2] = out_alpha_tile_gpu;
                        bindings[3] = out_gpu;

                        std::vector<ncnn::vk_constant_type> constants(16);
                        constants[0].i = in_gpu.w;
                        constants[1].i = in_gpu.h;
                        constants[2].i = in_gpu.cstep;
                        constants[3].i = out_tile_gpu.w;
                        constants[4].i = out_tile_gpu.h;
                        constants[5].i = out_tile_gpu.cstep;
                        constants[6].i = out_gpu.w;
                        constants[7].i = out_gpu.h;
                        constants[8].i = out_gpu.cstep;
                        constants[9].i = xi * tile_w;
                        constants[10].i = std::min(yi * tile_h, prepadding);
                        constants[11].i = xi * tile_w * scale;
                        constants[12].i = std::min(tile_w * scale, out_gpu.w - xi * tile_w * scale);
                        constants[13].i = channels;
                        constants[14].i = out_alpha_tile_gpu.w;
                        constants[15].i = out_alpha_tile_gpu.h;

                        ncnn::VkMat dispatcher;
                        dispatcher.w = std::min(tile_w * scale, out_gpu.w - xi * tile_w * scale);
                        dispatcher.h = out_gpu.h;
                        dispatcher.c = channels;

                        cmd.record_pipeline(realcugan_4x_postproc, bindings, constants, dispatcher);
                    }
                    else
                    {
                        std::vector<ncnn::VkMat> bindings(3);
                        bindings[0] = out_tile_gpu;
                        bindings[1] = out_alpha_tile_gpu;
                        bindings[2] = out_gpu;

                        std::vector<ncnn::vk_constant_type> constants(11);
                        constants[0].i = out_tile_gpu.w;
                        constants[1].i = out_tile_gpu.h;
                        constants[2].i = out_tile_gpu.cstep;
                        constants[3].i = out_gpu.w;
                        constants[4].i = out_gpu.h;
                        constants[5].i = out_gpu.cstep;
                        constants[6].i = xi * tile_w * scale;
                        constants[7].i = std::min(tile_w * scale, out_gpu.w - xi * tile_w * scale);
                        constants[8].i = channels;
                        constants[9].i = out_alpha_tile_gpu.w;
                        constants[10].i = out_alpha_tile_gpu.h;

                        ncnn::VkMat dispatcher;
                        dispatcher.w = std::min(tile_w * scale, out_gpu.w - xi * tile_w * scale);
                        dispatcher.h = out_gpu.h;
                        dispatcher.c = channels;

                        cmd.record_pipeline(realcugan_postproc, bindings, constants, dispatcher);
                    }
                }

                if (xtiles > 1)
                {
                    int ret = cmd.submit_and_wait();
                    if (ret != 0)
                    {
                        extract_failed = true;
                        extract_ret = ret;
                        break;
                    }
                    cmd.reset();
                }
            }

            if (extract_failed) break;

            // download
            {
                ncnn::Mat out;

                if ((opt.use_fp16_storage || opt.use_fp16_packed) && opt.use_int8_storage)
                {
                    out = ncnn::Mat(out_gpu.w, out_gpu.h, (unsigned char*)outimage.data + (size_t)out_tile_y0 * scale * w * scale * channels, (size_t)channels, 1);
                }

                cmd.record_clone(out_gpu, out, opt);

                int ret = cmd.submit_and_wait();
                if (ret != 0)
                {
                    extract_failed = true;
                    extract_ret = ret;
                    break;
                }

                if (!((opt.use_fp16_storage || opt.use_fp16_packed) && opt.use_int8_storage))
                {
                    if (channels == 3)
                    {
                        out.to_pixels((unsigned char*)outimage.data + (size_t)out_tile_y0 * scale * w * scale * channels, bgr_mode ? ncnn::Mat::PIXEL_RGB2BGR : ncnn::Mat::PIXEL_RGB);
                    }
                    if (channels == 4)
                    {
                        out.to_pixels((unsigned char*)outimage.data + (size_t)out_tile_y0 * scale * w * scale * channels, bgr_mode ? ncnn::Mat::PIXEL_RGBA2BGRA : ncnn::Mat::PIXEL_RGBA);
                    }
                }
            }
        }

        vkdev->reclaim_blob_allocator(blob_vkallocator);
        vkdev->reclaim_staging_allocator(staging_vkallocator);

        if (extract_failed)
        {
            if (can_single_tile)
            {
                single_tile_failed = true;
                fprintf(stderr, "warning: single-tile extract failed (error code %d), falling back to tiled processing\n", extract_ret);
                continue;
            }
            if (current_tilesize <= 32)
            {
                fprintf(stderr, "process failed even at tilesize 32 (error code %d)\n", extract_ret);
                return extract_ret;
            }
            int next_tilesize = std::max(32, (current_tilesize / 2 / 32) * 32);
            if (next_tilesize == current_tilesize) next_tilesize = 32;
            fprintf(stderr, "warning: extract failed (error code %d, possible VRAM OOM), halving tilesize from %d to %d and retrying\n", extract_ret, current_tilesize, next_tilesize);
            current_tilesize = next_tilesize;
            continue;
        }

        return 0;
    }

    return -1;
}

int RealCUGAN::process_cpu(const ncnn::Mat& inimage, ncnn::Mat& outimage) const
{
    if (noise == -1 && scale == 1)
    {
        if (outimage.data && inimage.data && outimage.data != inimage.data)
        {
            memcpy(outimage.data, inimage.data, (size_t)inimage.w * inimage.h * inimage.elempack);
        }
        else
        {
            outimage = inimage;
        }
        return 0;
    }

    const unsigned char* pixeldata = (const unsigned char*)inimage.data;
    const int w = inimage.w;
    const int h = inimage.h;
    const int channels = inimage.elempack;

    int current_tilesize = this->tilesize;
    int64_t padded_area = (int64_t)(w + 2 * prepadding) * (h + 2 * prepadding);
    int64_t max_tile_area = (int64_t)(current_tilesize + 2 * prepadding) * (current_tilesize + 2 * prepadding);
    bool can_single_tile = (current_tilesize > 0 && padded_area <= max_tile_area);

    int xtiles = can_single_tile ? 1 : ((w + current_tilesize - 1) / current_tilesize);
    int ytiles = can_single_tile ? 1 : ((h + current_tilesize - 1) / current_tilesize);
    const int tile_w = can_single_tile ? w : (((w + xtiles - 1) / xtiles + 3) & ~3);
    const int tile_h = can_single_tile ? h : (((h + ytiles - 1) / ytiles + 3) & ~3);
    xtiles = can_single_tile ? 1 : ((w + tile_w - 1) / tile_w);
    ytiles = can_single_tile ? 1 : ((h + tile_h - 1) / tile_h);

    ncnn::Option opt = net.opt;

    for (int yi = 0; yi < ytiles; yi++)
    {
        const int tile_h_nopad = std::min((yi + 1) * tile_h, h) - yi * tile_h;

        int prepadding_bottom = prepadding;
        if (scale == 1 || scale == 3)
        {
            prepadding_bottom += (tile_h_nopad + 3) / 4 * 4 - tile_h_nopad;
        }
        if (scale == 2 || scale == 4)
        {
            prepadding_bottom += (tile_h_nopad + 1) / 2 * 2 - tile_h_nopad;
        }

        int in_tile_y0 = std::max(yi * tile_h - prepadding, 0);
        int in_tile_y1 = std::min((yi + 1) * tile_h + prepadding_bottom, h);

        for (int xi = 0; xi < xtiles; xi++)
        {
            const int tile_w_nopad = std::min((xi + 1) * tile_w, w) - xi * tile_w;

            int prepadding_right = prepadding;
            if (scale == 1 || scale == 3)
            {
                prepadding_right += (tile_w_nopad + 3) / 4 * 4 - tile_w_nopad;
            }
            if (scale == 2 || scale == 4)
            {
                prepadding_right += (tile_w_nopad + 1) / 2 * 2 - tile_w_nopad;
            }

            int in_tile_x0 = std::max(xi * tile_w - prepadding, 0);
            int in_tile_x1 = std::min((xi + 1) * tile_w + prepadding_right, w);

            // crop tile
            ncnn::Mat in, in_nopad;
            {
                if (channels == 3)
                {
                    in = ncnn::Mat::from_pixels_roi(pixeldata, bgr_mode ? ncnn::Mat::PIXEL_BGR2RGB : ncnn::Mat::PIXEL_RGB, w, h, in_tile_x0, in_tile_y0, in_tile_x1 - in_tile_x0, in_tile_y1 - in_tile_y0);
                }
                if (channels == 4)
                {
                    in_nopad = ncnn::Mat::from_pixels_roi(pixeldata, bgr_mode ? ncnn::Mat::PIXEL_BGRA2RGBA : ncnn::Mat::PIXEL_RGBA, w, h, xi * tile_w, yi * tile_h, tile_w_nopad, tile_h_nopad);
                    in = ncnn::Mat::from_pixels_roi(pixeldata, bgr_mode ? ncnn::Mat::PIXEL_BGRA2RGBA : ncnn::Mat::PIXEL_RGBA, w, h, in_tile_x0, in_tile_y0, in_tile_x1 - in_tile_x0, in_tile_y1 - in_tile_y0);
                }
            }

            ncnn::Mat out;

            if (tta_level > 1)
            {
                // split alpha and preproc
                ncnn::Mat in_tile[8];
                ncnn::Mat in_alpha_tile;
                {
                    const float norm_scale = is_pro_model ? (0.7f / 255.f) : (1.f / 255.f);
                    const float norm_bias = is_pro_model ? 0.15f : 0.0f;

                    in_tile[0].create(in.w, in.h, 3);
                    for (int q = 0; q < 3; q++)
                    {
                        const float* ptr = in.channel(q);
                        float* outptr0 = in_tile[0].channel(q);

                        for (int i = 0; i < in.h; i++)
                        {
                            for (int j = 0; j < in.w; j++)
                            {
                                *outptr0++ = *ptr++ * norm_scale + norm_bias;
                            }
                        }
                    }

                    if (channels == 4)
                    {
                        in_alpha_tile = in_nopad.channel_range(3, 1).clone();
                    }
                }

                // border padding
                {
                    int pad_top = std::max(prepadding - yi * tile_h, 0);
                    int pad_bottom = std::max(std::min((yi + 1) * tile_h + prepadding_bottom - h, prepadding_bottom), 0);
                    int pad_left = std::max(prepadding - xi * tile_w, 0);
                    int pad_right = std::max(std::min((xi + 1) * tile_w + prepadding_right - w, prepadding_right), 0);

                    ncnn::Mat in_tile_padded;
                    ncnn::copy_make_border(in_tile[0], in_tile_padded, pad_top, pad_bottom, pad_left, pad_right, 2, 0.f, net.opt);
                    in_tile[0] = in_tile_padded;
                }

                // create symmetries up to tta_level
                in_tile[1].create(in_tile[0].w, in_tile[0].h, 3);
                if (tta_level >= 4)
                {
                    in_tile[2].create(in_tile[0].w, in_tile[0].h, 3);
                    in_tile[3].create(in_tile[0].w, in_tile[0].h, 3);
                }
                if (tta_level >= 8)
                {
                    in_tile[4].create(in_tile[0].h, in_tile[0].w, 3);
                    in_tile[5].create(in_tile[0].h, in_tile[0].w, 3);
                    in_tile[6].create(in_tile[0].h, in_tile[0].w, 3);
                    in_tile[7].create(in_tile[0].h, in_tile[0].w, 3);
                }

                for (int q = 0; q < 3; q++)
                {
                    const ncnn::Mat in_tile_0 = in_tile[0].channel(q);
                    ncnn::Mat in_tile_1 = in_tile[1].channel(q);
                    ncnn::Mat in_tile_2 = (tta_level >= 4) ? in_tile[2].channel(q) : ncnn::Mat();
                    ncnn::Mat in_tile_3 = (tta_level >= 4) ? in_tile[3].channel(q) : ncnn::Mat();
                    ncnn::Mat in_tile_4 = (tta_level >= 8) ? in_tile[4].channel(q) : ncnn::Mat();
                    ncnn::Mat in_tile_5 = (tta_level >= 8) ? in_tile[5].channel(q) : ncnn::Mat();
                    ncnn::Mat in_tile_6 = (tta_level >= 8) ? in_tile[6].channel(q) : ncnn::Mat();
                    ncnn::Mat in_tile_7 = (tta_level >= 8) ? in_tile[7].channel(q) : ncnn::Mat();

                    for (int i = 0; i < in_tile[0].h; i++)
                    {
                        const float* outptr0 = in_tile_0.row(i);
                        float* outptr1 = in_tile_1.row(i) + in_tile[0].w - 1;
                        float* outptr2 = (tta_level >= 4) ? (in_tile_2.row(in_tile[0].h - 1 - i) + in_tile[0].w - 1) : nullptr;
                        float* outptr3 = (tta_level >= 4) ? in_tile_3.row(in_tile[0].h - 1 - i) : nullptr;

                        for (int j = 0; j < in_tile[0].w; j++)
                        {
                            float v = *outptr0++;
                            *outptr1-- = v;
                            if (tta_level >= 4)
                            {
                                *outptr2-- = v;
                                *outptr3++ = v;
                            }
                            if (tta_level >= 8)
                            {
                                float* outptr4 = in_tile_4.row(j) + i;
                                float* outptr5 = in_tile_5.row(in_tile[0].w - 1 - j) + i;
                                float* outptr6 = in_tile_6.row(j) + in_tile[0].h - 1 - i;
                                float* outptr7 = in_tile_7.row(in_tile[0].w - 1 - j) + in_tile[0].h - 1 - i;
                                *outptr4 = v;
                                *outptr5 = v;
                                *outptr6 = v;
                                *outptr7 = v;
                            }
                        }
                    }
                }

                // realcugan
                ncnn::Mat out_tile[8];
                for (int ti = 0; ti < tta_level; ti++)
                {
                    ncnn::Extractor ex = net.create_extractor();
                    ex.input("in0", in_tile[ti]);
                    ex.extract("out0", out_tile[ti]);
                }

                ncnn::Mat out_alpha_tile;
                if (channels == 4)
                {
                    if (scale == 1)
                    {
                        out_alpha_tile = in_alpha_tile;
                    }
                    if (scale == 2)
                    {
                        bicubic_2x->forward(in_alpha_tile, out_alpha_tile, opt);
                    }
                    if (scale == 3)
                    {
                        bicubic_3x->forward(in_alpha_tile, out_alpha_tile, opt);
                    }
                    if (scale == 4)
                    {
                        bicubic_4x->forward(in_alpha_tile, out_alpha_tile, opt);
                    }
                }

                // postproc and merge alpha
                {
                    out.create(tile_w_nopad * scale, tile_h_nopad * scale, channels);
                    if (scale == 4)
                    {
                        for (int q = 0; q < 3; q++)
                        {
                            const ncnn::Mat out_tile_0 = out_tile[0].channel(q);
                            const ncnn::Mat out_tile_1 = out_tile[1].channel(q);
                            const ncnn::Mat out_tile_2 = (tta_level >= 4) ? out_tile[2].channel(q) : ncnn::Mat();
                            const ncnn::Mat out_tile_3 = (tta_level >= 4) ? out_tile[3].channel(q) : ncnn::Mat();
                            const ncnn::Mat out_tile_4 = (tta_level >= 8) ? out_tile[4].channel(q) : ncnn::Mat();
                            const ncnn::Mat out_tile_5 = (tta_level >= 8) ? out_tile[5].channel(q) : ncnn::Mat();
                            const ncnn::Mat out_tile_6 = (tta_level >= 8) ? out_tile[6].channel(q) : ncnn::Mat();
                            const ncnn::Mat out_tile_7 = (tta_level >= 8) ? out_tile[7].channel(q) : ncnn::Mat();
                            float* outptr = out.channel(q);

                            for (int i = 0; i < out.h; i++)
                            {
                                const float* inptr = in_tile[0].channel(q).row(prepadding + i / 4) + prepadding;
                                const float* ptr0 = out_tile_0.row(i);
                                const float* ptr1 = out_tile_1.row(i) + out_tile[0].w - 1;
                                const float* ptr2 = (tta_level >= 4) ? (out_tile_2.row(out_tile[0].h - 1 - i) + out_tile[0].w - 1) : nullptr;
                                const float* ptr3 = (tta_level >= 4) ? out_tile_3.row(out_tile[0].h - 1 - i) : nullptr;

                                for (int j = 0; j < out.w; j++)
                                {
                                    float v0 = *ptr0++;
                                    float v1 = *ptr1--;
                                    float v = 0.f;
                                    if (tta_level == 2)
                                    {
                                        v = (v0 + v1) * 0.5f;
                                    }
                                    else if (tta_level == 4)
                                    {
                                        float v2 = *ptr2--;
                                        float v3 = *ptr3++;
                                        v = (v0 + v1 + v2 + v3) * 0.25f;
                                    }
                                    else
                                    {
                                        float v2 = *ptr2--;
                                        float v3 = *ptr3++;
                                        float v4 = *(out_tile_4.row(j) + i);
                                        float v5 = *(out_tile_5.row(out_tile[0].w - 1 - j) + i);
                                        float v6 = *(out_tile_6.row(j) + out_tile[0].h - 1 - i);
                                        float v7 = *(out_tile_7.row(out_tile[0].w - 1 - j) + out_tile[0].h - 1 - i);
                                        v = (v0 + v1 + v2 + v3 + v4 + v5 + v6 + v7) * 0.125f;
                                    }

                                    if (is_pro_model)
                                    {
                                        v = (v + inptr[j / 4] - 0.15f) * (255.f / 0.7f);
                                    }
                                    else
                                    {
                                        v = v * 255.f + 0.5f + inptr[j / 4] * 255.f;
                                    }
                                    *outptr++ = std::max(0.f, std::min(255.f, v));
                                }
                            }
                        }
                    }
                    else
                    {
                        for (int q = 0; q < 3; q++)
                        {
                            const ncnn::Mat out_tile_0 = out_tile[0].channel(q);
                            const ncnn::Mat out_tile_1 = out_tile[1].channel(q);
                            const ncnn::Mat out_tile_2 = (tta_level >= 4) ? out_tile[2].channel(q) : ncnn::Mat();
                            const ncnn::Mat out_tile_3 = (tta_level >= 4) ? out_tile[3].channel(q) : ncnn::Mat();
                            const ncnn::Mat out_tile_4 = (tta_level >= 8) ? out_tile[4].channel(q) : ncnn::Mat();
                            const ncnn::Mat out_tile_5 = (tta_level >= 8) ? out_tile[5].channel(q) : ncnn::Mat();
                            const ncnn::Mat out_tile_6 = (tta_level >= 8) ? out_tile[6].channel(q) : ncnn::Mat();
                            const ncnn::Mat out_tile_7 = (tta_level >= 8) ? out_tile[7].channel(q) : ncnn::Mat();
                            float* outptr = out.channel(q);

                            for (int i = 0; i < out.h; i++)
                            {
                                const float* ptr0 = out_tile_0.row(i);
                                const float* ptr1 = out_tile_1.row(i) + out_tile[0].w - 1;
                                const float* ptr2 = (tta_level >= 4) ? (out_tile_2.row(out_tile[0].h - 1 - i) + out_tile[0].w - 1) : nullptr;
                                const float* ptr3 = (tta_level >= 4) ? out_tile_3.row(out_tile[0].h - 1 - i) : nullptr;

                                for (int j = 0; j < out.w; j++)
                                {
                                    float v0 = *ptr0++;
                                    float v1 = *ptr1--;
                                    float v = 0.f;
                                    if (tta_level == 2)
                                    {
                                        v = (v0 + v1) * 0.5f;
                                    }
                                    else if (tta_level == 4)
                                    {
                                        float v2 = *ptr2--;
                                        float v3 = *ptr3++;
                                        v = (v0 + v1 + v2 + v3) * 0.25f;
                                    }
                                    else
                                    {
                                        float v2 = *ptr2--;
                                        float v3 = *ptr3++;
                                        float v4 = *(out_tile_4.row(j) + i);
                                        float v5 = *(out_tile_5.row(out_tile[0].w - 1 - j) + i);
                                        float v6 = *(out_tile_6.row(j) + out_tile[0].h - 1 - i);
                                        float v7 = *(out_tile_7.row(out_tile[0].w - 1 - j) + out_tile[0].h - 1 - i);
                                        v = (v0 + v1 + v2 + v3 + v4 + v5 + v6 + v7) * 0.125f;
                                    }

                                    if (is_pro_model)
                                        v = (v - 0.15f) * (255.f / 0.7f);
                                    else
                                        v = v * 255.f;

                                    *outptr++ = std::max(0.f, std::min(255.f, v + 0.5f));
                                }
                            }
                        }
                    }

                    if (channels == 4)
                    {
                        memcpy(out.channel_range(3, 1), out_alpha_tile, out_alpha_tile.total() * sizeof(float));
                    }
                }
            }
            else
            {
                // split alpha and preproc
                ncnn::Mat in_tile;
                ncnn::Mat in_alpha_tile;
                {
                    in_tile.create(in.w, in.h, 3);
                    const float norm_scale = is_pro_model ? (0.7f / 255.f) : (1.f / 255.f);
                    const float norm_bias = is_pro_model ? 0.15f : 0.0f;

                    for (int q = 0; q < 3; q++)
                    {
                        const float* ptr = in.channel(q);
                        float* outptr = in_tile.channel(q);

                        for (int i = 0; i < in.w * in.h; i++)
                        {
                            *outptr++ = *ptr++ * norm_scale + norm_bias;
                        }
                    }

                    if (channels == 4)
                    {
                        in_alpha_tile = in_nopad.channel_range(3, 1).clone();
                    }
                }

                // border padding
                {
                    int pad_top = std::max(prepadding - yi * tile_h, 0);
                    int pad_bottom = std::max(std::min((yi + 1) * tile_h + prepadding_bottom - h, prepadding_bottom), 0);
                    int pad_left = std::max(prepadding - xi * tile_w, 0);
                    int pad_right = std::max(std::min((xi + 1) * tile_w + prepadding_right - w, prepadding_right), 0);

                    ncnn::Mat in_tile_padded;
                    ncnn::copy_make_border(in_tile, in_tile_padded, pad_top, pad_bottom, pad_left, pad_right, 2, 0.f, net.opt);
                    in_tile = in_tile_padded;
                }

                // realcugan
                ncnn::Mat out_tile;
                {
                    ncnn::Extractor ex = net.create_extractor();

                    ex.input("in0", in_tile);

                    ex.extract("out0", out_tile);
                }

                ncnn::Mat out_alpha_tile;
                if (channels == 4)
                {
                    if (scale == 1)
                    {
                        out_alpha_tile = in_alpha_tile;
                    }
                    if (scale == 2)
                    {
                        bicubic_2x->forward(in_alpha_tile, out_alpha_tile, opt);
                    }
                    if (scale == 3)
                    {
                        bicubic_3x->forward(in_alpha_tile, out_alpha_tile, opt);
                    }
                    if (scale == 4)
                    {
                        bicubic_4x->forward(in_alpha_tile, out_alpha_tile, opt);
                    }
                }

                // postproc and merge alpha
                {
                    out.create(tile_w_nopad * scale, tile_h_nopad * scale, channels);
                    if (scale == 4)
                    {
                        for (int q = 0; q < 3; q++)
                        {
                            float* outptr = out.channel(q);

                            for (int i = 0; i < out.h; i++)
                            {
                                const float* inptr = in_tile.channel(q).row(prepadding + i / 4) + prepadding;
                                const float* ptr = out_tile.channel(q).row(i);

                                for (int j = 0; j < out.w; j++)
                                {
                                    float v = *ptr++;
                                    if (is_pro_model)
                                    {
                                        v = (v + inptr[j / 4] - 0.15f) * (255.f / 0.7f);
                                    }
                                    else
                                    {
                                        v = v * 255.f + 0.5f + inptr[j / 4] * 255.f;
                                    }
                                    *outptr++ = std::max(0.f, std::min(255.f, v));
                                }
                            }
                        }
                    }
                    else
                    {
                        for (int q = 0; q < 3; q++)
                        {
                            float* outptr = out.channel(q);

                            for (int i = 0; i < out.h; i++)
                            {
                                const float* ptr = out_tile.channel(q).row(i);

                                for (int j = 0; j < out.w; j++)
                                {
                                    float v = *ptr++;
                                    if (is_pro_model)
                                        v = (v - 0.15f) * (255.f / 0.7f);
                                    else
                                        v = v * 255.f;

                                    *outptr++ = std::max(0.f, std::min(255.f, v + 0.5f));
                                }
                            }
                        }
                    }

                    if (channels == 4)
                    {
                        memcpy(out.channel_range(3, 1), out_alpha_tile, out_alpha_tile.total() * sizeof(float));
                    }
                }
            }

            {
                if (channels == 3)
                {
                    out.to_pixels((unsigned char*)outimage.data + (size_t)yi * scale * tile_h * w * scale * channels + (size_t)xi * scale * tile_w * channels, bgr_mode ? ncnn::Mat::PIXEL_RGB2BGR : ncnn::Mat::PIXEL_RGB, w * scale * channels);
                }
                if (channels == 4)
                {
                    out.to_pixels((unsigned char*)outimage.data + (size_t)yi * scale * tile_h * w * scale * channels + (size_t)xi * scale * tile_w * channels, bgr_mode ? ncnn::Mat::PIXEL_RGBA2BGRA : ncnn::Mat::PIXEL_RGBA, w * scale * channels);
                }
            }
        }
    }

    return 0;
}

int RealCUGAN::process_se(const ncnn::Mat& inimage, ncnn::Mat& outimage, int current_tilesize) const
{
    ncnn::VkAllocator* blob_vkallocator = vkdev->acquire_blob_allocator();
    ncnn::VkAllocator* staging_vkallocator = vkdev->acquire_staging_allocator();

    ncnn::Option opt = net.opt;
    opt.blob_vkallocator = blob_vkallocator;
    opt.workspace_vkallocator = blob_vkallocator;
    opt.staging_vkallocator = staging_vkallocator;

    FeatureCache cache;

    int ret = 0;
    std::vector<std::string> in0 = {};
    std::vector<std::string> out0 = {"gap0"};
    ret = process_se_stage0(inimage, in0, out0, opt, cache, current_tilesize);
    if (ret != 0) goto cleanup;

    {
        std::vector<std::string> gap0 = {"gap0"};
        ret = process_se_sync_gap(inimage, gap0, opt, cache, current_tilesize);
        if (ret != 0) goto cleanup;
    }

    {
        std::vector<std::string> in1 = {"gap0"};
        std::vector<std::string> out1 = {"gap1"};
        ret = process_se_stage0(inimage, in1, out1, opt, cache, current_tilesize);
        if (ret != 0) goto cleanup;
    }

    {
        std::vector<std::string> gap1 = {"gap1"};
        ret = process_se_sync_gap(inimage, gap1, opt, cache, current_tilesize);
        if (ret != 0) goto cleanup;
    }

    {
        std::vector<std::string> in2 = {"gap0", "gap1"};
        std::vector<std::string> out2 = {"gap2"};
        ret = process_se_stage0(inimage, in2, out2, opt, cache, current_tilesize);
        if (ret != 0) goto cleanup;
    }

    {
        std::vector<std::string> gap2 = {"gap2"};
        ret = process_se_sync_gap(inimage, gap2, opt, cache, current_tilesize);
        if (ret != 0) goto cleanup;
    }

    {
        std::vector<std::string> in3 = {"gap0", "gap1", "gap2"};
        std::vector<std::string> out3 = {"gap3"};
        ret = process_se_stage0(inimage, in3, out3, opt, cache, current_tilesize);
        if (ret != 0) goto cleanup;
    }

    {
        std::vector<std::string> gap3 = {"gap3"};
        ret = process_se_sync_gap(inimage, gap3, opt, cache, current_tilesize);
        if (ret != 0) goto cleanup;
    }

    {
        std::vector<std::string> in4 = {"gap0", "gap1", "gap2", "gap3"};
        ret = process_se_stage2(inimage, in4, outimage, opt, cache, current_tilesize);
    }

cleanup:
    cache.clear();

    vkdev->reclaim_blob_allocator(blob_vkallocator);
    vkdev->reclaim_staging_allocator(staging_vkallocator);

    return ret;
}

int RealCUGAN::process_se_rough(const ncnn::Mat& inimage, ncnn::Mat& outimage, int current_tilesize) const
{
    ncnn::VkAllocator* blob_vkallocator = vkdev->acquire_blob_allocator();
    ncnn::VkAllocator* staging_vkallocator = vkdev->acquire_staging_allocator();

    ncnn::Option opt = net.opt;
    opt.blob_vkallocator = blob_vkallocator;
    opt.workspace_vkallocator = blob_vkallocator;
    opt.staging_vkallocator = staging_vkallocator;

    FeatureCache cache;

    int ret = 0;
    std::vector<std::string> in0 = {};
    std::vector<std::string> out0 = {"gap0", "gap1", "gap2", "gap3"};
    ret = process_se_stage0(inimage, in0, out0, opt, cache, current_tilesize);
    if (ret != 0) goto cleanup;

    {
        std::vector<std::string> gap0 = {"gap0", "gap1", "gap2", "gap3"};
        ret = process_se_sync_gap(inimage, gap0, opt, cache, current_tilesize);
        if (ret != 0) goto cleanup;
    }

    {
        std::vector<std::string> in4 = {"gap0", "gap1", "gap2", "gap3"};
        ret = process_se_stage2(inimage, in4, outimage, opt, cache, current_tilesize);
    }

cleanup:
    cache.clear();

    vkdev->reclaim_blob_allocator(blob_vkallocator);
    vkdev->reclaim_staging_allocator(staging_vkallocator);

    return ret;
}

int RealCUGAN::process_se_very_rough(const ncnn::Mat& inimage, ncnn::Mat& outimage, int current_tilesize) const
{
    ncnn::VkAllocator* blob_vkallocator = vkdev->acquire_blob_allocator();
    ncnn::VkAllocator* staging_vkallocator = vkdev->acquire_staging_allocator();

    ncnn::Option opt = net.opt;
    opt.blob_vkallocator = blob_vkallocator;
    opt.workspace_vkallocator = blob_vkallocator;
    opt.staging_vkallocator = staging_vkallocator;

    FeatureCache cache;

    int ret = 0;
    std::vector<std::string> in0 = {};
    std::vector<std::string> out0 = {"gap0", "gap1", "gap2", "gap3"};
    ret = process_se_very_rough_stage0(inimage, in0, out0, opt, cache, current_tilesize);
    if (ret != 0) goto cleanup;

    {
        std::vector<std::string> gap0 = {"gap0", "gap1", "gap2", "gap3"};
        ret = process_se_very_rough_sync_gap(inimage, gap0, opt, cache, current_tilesize);
        if (ret != 0) goto cleanup;
    }

    {
        std::vector<std::string> in4 = {"gap0", "gap1", "gap2", "gap3"};
        ret = process_se_stage2(inimage, in4, outimage, opt, cache, current_tilesize);
    }

cleanup:
    cache.clear();

    vkdev->reclaim_blob_allocator(blob_vkallocator);
    vkdev->reclaim_staging_allocator(staging_vkallocator);

    return ret;
}

int RealCUGAN::process_cpu_se(const ncnn::Mat& inimage, ncnn::Mat& outimage, int current_tilesize) const
{
    FeatureCache cache;

    int ret = 0;
    std::vector<std::string> in0 = {};
    std::vector<std::string> out0 = {"gap0"};
    ret = process_cpu_se_stage0(inimage, in0, out0, cache, current_tilesize);
    if (ret != 0) { cache.clear(); return ret; }

    {
        std::vector<std::string> gap0 = {"gap0"};
        ret = process_cpu_se_sync_gap(inimage, gap0, cache, current_tilesize);
        if (ret != 0) { cache.clear(); return ret; }
    }

    {
        std::vector<std::string> in1 = {"gap0"};
        std::vector<std::string> out1 = {"gap1"};
        ret = process_cpu_se_stage0(inimage, in1, out1, cache, current_tilesize);
        if (ret != 0) { cache.clear(); return ret; }
    }

    {
        std::vector<std::string> gap1 = {"gap1"};
        ret = process_cpu_se_sync_gap(inimage, gap1, cache, current_tilesize);
        if (ret != 0) { cache.clear(); return ret; }
    }

    {
        std::vector<std::string> in2 = {"gap0", "gap1"};
        std::vector<std::string> out2 = {"gap2"};
        ret = process_cpu_se_stage0(inimage, in2, out2, cache, current_tilesize);
        if (ret != 0) { cache.clear(); return ret; }
    }

    {
        std::vector<std::string> gap2 = {"gap2"};
        ret = process_cpu_se_sync_gap(inimage, gap2, cache, current_tilesize);
        if (ret != 0) { cache.clear(); return ret; }
    }

    {
        std::vector<std::string> in3 = {"gap0", "gap1", "gap2"};
        std::vector<std::string> out3 = {"gap3"};
        ret = process_cpu_se_stage0(inimage, in3, out3, cache, current_tilesize);
        if (ret != 0) { cache.clear(); return ret; }
    }

    {
        std::vector<std::string> gap3 = {"gap3"};
        ret = process_cpu_se_sync_gap(inimage, gap3, cache, current_tilesize);
        if (ret != 0) { cache.clear(); return ret; }
    }

    {
        std::vector<std::string> in4 = {"gap0", "gap1", "gap2", "gap3"};
        ret = process_cpu_se_stage2(inimage, in4, outimage, cache, current_tilesize);
    }

    cache.clear();

    return ret;
}

int RealCUGAN::process_cpu_se_rough(const ncnn::Mat& inimage, ncnn::Mat& outimage, int current_tilesize) const
{
    FeatureCache cache;

    int ret = 0;
    std::vector<std::string> in0 = {};
    std::vector<std::string> out0 = {"gap0", "gap1", "gap2", "gap3"};
    ret = process_cpu_se_stage0(inimage, in0, out0, cache, current_tilesize);
    if (ret != 0) { cache.clear(); return ret; }

    {
        std::vector<std::string> gap0 = {"gap0", "gap1", "gap2", "gap3"};
        ret = process_cpu_se_sync_gap(inimage, gap0, cache, current_tilesize);
        if (ret != 0) { cache.clear(); return ret; }
    }

    {
        std::vector<std::string> in4 = {"gap0", "gap1", "gap2", "gap3"};
        ret = process_cpu_se_stage2(inimage, in4, outimage, cache, current_tilesize);
    }

    cache.clear();

    return ret;
}

int RealCUGAN::process_cpu_se_very_rough(const ncnn::Mat& inimage, ncnn::Mat& outimage, int current_tilesize) const
{
    FeatureCache cache;

    int ret = 0;
    std::vector<std::string> in0 = {};
    std::vector<std::string> out0 = {"gap0", "gap1", "gap2", "gap3"};
    ret = process_cpu_se_very_rough_stage0(inimage, in0, out0, cache, current_tilesize);
    if (ret != 0) { cache.clear(); return ret; }

    {
        std::vector<std::string> gap0 = {"gap0", "gap1", "gap2", "gap3"};
        ret = process_cpu_se_very_rough_sync_gap(inimage, gap0, cache, current_tilesize);
        if (ret != 0) { cache.clear(); return ret; }
    }

    {
        std::vector<std::string> in4 = {"gap0", "gap1", "gap2", "gap3"};
        ret = process_cpu_se_stage2(inimage, in4, outimage, cache, current_tilesize);
    }

    cache.clear();

    return ret;
}

int RealCUGAN::process_se_stage0(const ncnn::Mat& inimage, const std::vector<std::string>& names, const std::vector<std::string>& outnames, const ncnn::Option& opt, FeatureCache& cache, int current_tilesize) const
{
    const unsigned char* pixeldata = (const unsigned char*)inimage.data;
    const int w = inimage.w;
    const int h = inimage.h;
    const int channels = inimage.elempack;

    const int TILE_SIZE_X = current_tilesize > 0 ? current_tilesize : tilesize;
    const int TILE_SIZE_Y = current_tilesize > 0 ? current_tilesize : tilesize;

    // each tile 400x400
    const int xtiles = (w + TILE_SIZE_X - 1) / TILE_SIZE_X;
    const int ytiles = (h + TILE_SIZE_Y - 1) / TILE_SIZE_Y;

    const size_t in_out_tile_elemsize = (opt.use_fp16_storage || opt.use_fp16_packed) ? 2u : 4u;

    //#pragma omp parallel for num_threads(2)
    for (int yi = 0; yi < ytiles; yi++)
    {
        const int tile_h_nopad = std::min((yi + 1) * TILE_SIZE_Y, h) - yi * TILE_SIZE_Y;

        int prepadding_bottom = prepadding;
        if (scale == 1 || scale == 3)
        {
            prepadding_bottom += (tile_h_nopad + 3) / 4 * 4 - tile_h_nopad;
        }
        if (scale == 2 || scale == 4)
        {
            prepadding_bottom += (tile_h_nopad + 1) / 2 * 2 - tile_h_nopad;
        }

        int in_tile_y0 = std::max(yi * TILE_SIZE_Y - prepadding, 0);
        int in_tile_y1 = std::min((yi + 1) * TILE_SIZE_Y + prepadding_bottom, h);

        ncnn::Mat in;
        if ((opt.use_fp16_storage || opt.use_fp16_packed) && opt.use_int8_storage)
        {
            in = ncnn::Mat(w, (in_tile_y1 - in_tile_y0), (unsigned char*)pixeldata + (size_t)in_tile_y0 * w * channels, (size_t)channels, 1);
        }
        else
        {
            if (channels == 3)
            {
                in = ncnn::Mat::from_pixels(pixeldata + (size_t)in_tile_y0 * w * channels, bgr_mode ? ncnn::Mat::PIXEL_BGR2RGB : ncnn::Mat::PIXEL_RGB, w, (in_tile_y1 - in_tile_y0));
            }
            if (channels == 4)
            {
                in = ncnn::Mat::from_pixels(pixeldata + (size_t)in_tile_y0 * w * channels, bgr_mode ? ncnn::Mat::PIXEL_BGRA2RGBA : ncnn::Mat::PIXEL_RGBA, w, (in_tile_y1 - in_tile_y0));
            }
        }

        ncnn::VkCompute cmd(vkdev);

        // upload
        ncnn::VkMat in_gpu;
        {
            cmd.record_clone(in, in_gpu, opt);

            if (xtiles > 1)
            {
                cmd.submit_and_wait();
                cmd.reset();
            }
        }

        for (int xi = 0; xi < xtiles; xi++)
        {
            const int tile_w_nopad = std::min((xi + 1) * TILE_SIZE_X, w) - xi * TILE_SIZE_X;

            int prepadding_right = prepadding;
            if (scale == 1 || scale == 3)
            {
                prepadding_right += (tile_w_nopad + 3) / 4 * 4 - tile_w_nopad;
            }
            if (scale == 2 || scale == 4)
            {
                prepadding_right += (tile_w_nopad + 1) / 2 * 2 - tile_w_nopad;
            }

            if (tta_level > 1)
            {
                // preproc
                ncnn::VkMat in_tile_gpu[8];
                ncnn::VkMat in_alpha_tile_gpu;
                ncnn::VkMat dummy_tile_gpu;
                if (tta_level < 8)
                {
                    dummy_tile_gpu.create(1, 1, 3, in_out_tile_elemsize, 1, opt.blob_vkallocator);
                }
                {
                    // crop tile
                    int tile_x0 = xi * TILE_SIZE_X - prepadding;
                    int tile_x1 = std::min((xi + 1) * TILE_SIZE_X, w) + prepadding_right;
                    int tile_y0 = yi * TILE_SIZE_Y - prepadding;
                    int tile_y1 = std::min((yi + 1) * TILE_SIZE_Y, h) + prepadding_bottom;

                    in_tile_gpu[0].create(tile_x1 - tile_x0, tile_y1 - tile_y0, 3, in_out_tile_elemsize, 1, opt.blob_vkallocator);
                    in_tile_gpu[1].create(tile_x1 - tile_x0, tile_y1 - tile_y0, 3, in_out_tile_elemsize, 1, opt.blob_vkallocator);
                    if (tta_level >= 4)
                    {
                        in_tile_gpu[2].create(tile_x1 - tile_x0, tile_y1 - tile_y0, 3, in_out_tile_elemsize, 1, opt.blob_vkallocator);
                        in_tile_gpu[3].create(tile_x1 - tile_x0, tile_y1 - tile_y0, 3, in_out_tile_elemsize, 1, opt.blob_vkallocator);
                    }
                    if (tta_level >= 8)
                    {
                        in_tile_gpu[4].create(tile_y1 - tile_y0, tile_x1 - tile_x0, 3, in_out_tile_elemsize, 1, opt.blob_vkallocator);
                        in_tile_gpu[5].create(tile_y1 - tile_y0, tile_x1 - tile_x0, 3, in_out_tile_elemsize, 1, opt.blob_vkallocator);
                        in_tile_gpu[6].create(tile_y1 - tile_y0, tile_x1 - tile_x0, 3, in_out_tile_elemsize, 1, opt.blob_vkallocator);
                        in_tile_gpu[7].create(tile_y1 - tile_y0, tile_x1 - tile_x0, 3, in_out_tile_elemsize, 1, opt.blob_vkallocator);
                    }

                    if (channels == 4)
                    {
                        in_alpha_tile_gpu.create(tile_w_nopad, tile_h_nopad, 1, in_out_tile_elemsize, 1, opt.blob_vkallocator);
                    }

                    std::vector<ncnn::VkMat> bindings(10);
                    bindings[0] = in_gpu;
                    for (int ti = 0; ti < tta_level; ti++)
                    {
                        bindings[1 + ti] = in_tile_gpu[ti];
                    }
                    for (int ti = tta_level; ti < 8; ti++)
                    {
                        bindings[1 + ti] = dummy_tile_gpu;
                    }
                    bindings[9] = in_alpha_tile_gpu;

                    std::vector<ncnn::vk_constant_type> constants(13);
                    constants[0].i = in_gpu.w;
                    constants[1].i = in_gpu.h;
                    constants[2].i = in_gpu.cstep;
                    constants[3].i = in_tile_gpu[0].w;
                    constants[4].i = in_tile_gpu[0].h;
                    constants[5].i = in_tile_gpu[0].cstep;
                    constants[6].i = prepadding;
                    constants[7].i = prepadding;
                    constants[8].i = xi * TILE_SIZE_X;
                    constants[9].i = std::min(yi * TILE_SIZE_Y, prepadding);
                    constants[10].i = channels;
                    constants[11].i = in_alpha_tile_gpu.w;
                    constants[12].i = in_alpha_tile_gpu.h;

                    ncnn::VkMat dispatcher;
                    dispatcher.w = in_tile_gpu[0].w;
                    dispatcher.h = in_tile_gpu[0].h;
                    dispatcher.c = channels;

                    cmd.record_pipeline(realcugan_preproc, bindings, constants, dispatcher);
                }

                // realcugan
                int tta_count = (syncgap >= 2) ? 1 : tta_level;
                for (int ti = 0; ti < tta_count; ti++)
                {
                    ncnn::Extractor ex = net.create_extractor();

                    ex.set_blob_vkallocator(opt.blob_vkallocator);
                    ex.set_workspace_vkallocator(opt.blob_vkallocator);
                    ex.set_staging_vkallocator(opt.staging_vkallocator);

                    ex.input("in0", in_tile_gpu[ti]);

                    for (size_t i = 0; i < names.size(); i++)
                    {
                        ncnn::VkMat feat;
                        cache.load(yi, xi, ti, names[i], feat);

                        ex.input(names[i].c_str(), feat);
                    }

                    for (size_t i = 0; i < outnames.size(); i++)
                    {
                        ncnn::VkMat feat;
                        int ret = ex.extract(outnames[i].c_str(), feat, cmd);
                        if (ret != 0) return ret;

                        cache.save(yi, xi, ti, outnames[i], feat);
                    }
                }
            }
            else
            {
                // preproc
                ncnn::VkMat in_tile_gpu;
                ncnn::VkMat in_alpha_tile_gpu;
                {
                    // crop tile
                    int tile_x0 = xi * TILE_SIZE_X - prepadding;
                    int tile_x1 = std::min((xi + 1) * TILE_SIZE_X, w) + prepadding_right;
                    int tile_y0 = yi * TILE_SIZE_Y - prepadding;
                    int tile_y1 = std::min((yi + 1) * TILE_SIZE_Y, h) + prepadding_bottom;

                    in_tile_gpu.create(tile_x1 - tile_x0, tile_y1 - tile_y0, 3, in_out_tile_elemsize, 1, opt.blob_vkallocator);

                    if (channels == 4)
                    {
                        in_alpha_tile_gpu.create(tile_w_nopad, tile_h_nopad, 1, in_out_tile_elemsize, 1, opt.blob_vkallocator);
                    }

                    std::vector<ncnn::VkMat> bindings(3);
                    bindings[0] = in_gpu;
                    bindings[1] = in_tile_gpu;
                    bindings[2] = in_alpha_tile_gpu;

                    std::vector<ncnn::vk_constant_type> constants(13);
                    constants[0].i = in_gpu.w;
                    constants[1].i = in_gpu.h;
                    constants[2].i = in_gpu.cstep;
                    constants[3].i = in_tile_gpu.w;
                    constants[4].i = in_tile_gpu.h;
                    constants[5].i = in_tile_gpu.cstep;
                    constants[6].i = prepadding;
                    constants[7].i = prepadding;
                    constants[8].i = xi * TILE_SIZE_X;
                    constants[9].i = std::min(yi * TILE_SIZE_Y, prepadding);
                    constants[10].i = channels;
                    constants[11].i = in_alpha_tile_gpu.w;
                    constants[12].i = in_alpha_tile_gpu.h;

                    ncnn::VkMat dispatcher;
                    dispatcher.w = in_tile_gpu.w;
                    dispatcher.h = in_tile_gpu.h;
                    dispatcher.c = channels;

                    cmd.record_pipeline(realcugan_preproc, bindings, constants, dispatcher);
                }

                // realcugan
                {
                    ncnn::Extractor ex = net.create_extractor();

                    ex.set_blob_vkallocator(opt.blob_vkallocator);
                    ex.set_workspace_vkallocator(opt.blob_vkallocator);
                    ex.set_staging_vkallocator(opt.staging_vkallocator);

                    ex.input("in0", in_tile_gpu);

                    for (size_t i = 0; i < names.size(); i++)
                    {
                        ncnn::VkMat feat;
                        cache.load(yi, xi, 0, names[i], feat);

                        ex.input(names[i].c_str(), feat);
                    }

                    for (size_t i = 0; i < outnames.size(); i++)
                    {
                        ncnn::VkMat feat;
                        int ret = ex.extract(outnames[i].c_str(), feat, cmd);
                        if (ret != 0) return ret;

                        cache.save(yi, xi, 0, outnames[i], feat);
                    }
                }
            }

            if (xtiles > 1)
            {
                cmd.submit_and_wait();
                cmd.reset();
            }
        }

        if (xtiles <= 1)
        {
            cmd.submit_and_wait();
            cmd.reset();
        }
    }

    return 0;
}

int RealCUGAN::process_se_stage2(const ncnn::Mat& inimage, const std::vector<std::string>& names, ncnn::Mat& outimage, const ncnn::Option& opt, FeatureCache& cache, int current_tilesize) const
{
    const unsigned char* pixeldata = (const unsigned char*)inimage.data;
    const int w = inimage.w;
    const int h = inimage.h;
    const int channels = inimage.elempack;

    const int TILE_SIZE_X = current_tilesize > 0 ? current_tilesize : tilesize;
    const int TILE_SIZE_Y = current_tilesize > 0 ? current_tilesize : tilesize;

    // each tile 400x400
    const int xtiles = (w + TILE_SIZE_X - 1) / TILE_SIZE_X;
    const int ytiles = (h + TILE_SIZE_Y - 1) / TILE_SIZE_Y;

    const size_t in_out_tile_elemsize = (opt.use_fp16_storage || opt.use_fp16_packed) ? 2u : 4u;

    //#pragma omp parallel for num_threads(2)
    for (int yi = 0; yi < ytiles; yi++)
    {
        const int tile_h_nopad = std::min((yi + 1) * TILE_SIZE_Y, h) - yi * TILE_SIZE_Y;

        int prepadding_bottom = prepadding;
        if (scale == 1 || scale == 3)
        {
            prepadding_bottom += (tile_h_nopad + 3) / 4 * 4 - tile_h_nopad;
        }
        if (scale == 2 || scale == 4)
        {
            prepadding_bottom += (tile_h_nopad + 1) / 2 * 2 - tile_h_nopad;
        }

        int in_tile_y0 = std::max(yi * TILE_SIZE_Y - prepadding, 0);
        int in_tile_y1 = std::min((yi + 1) * TILE_SIZE_Y + prepadding_bottom, h);

        ncnn::Mat in;
        if ((opt.use_fp16_storage || opt.use_fp16_packed) && opt.use_int8_storage)
        {
            in = ncnn::Mat(w, (in_tile_y1 - in_tile_y0), (unsigned char*)pixeldata + (size_t)in_tile_y0 * w * channels, (size_t)channels, 1);
        }
        else
        {
            if (channels == 3)
            {
                in = ncnn::Mat::from_pixels(pixeldata + (size_t)in_tile_y0 * w * channels, bgr_mode ? ncnn::Mat::PIXEL_BGR2RGB : ncnn::Mat::PIXEL_RGB, w, (in_tile_y1 - in_tile_y0));
            }
            if (channels == 4)
            {
                in = ncnn::Mat::from_pixels(pixeldata + (size_t)in_tile_y0 * w * channels, bgr_mode ? ncnn::Mat::PIXEL_BGRA2RGBA : ncnn::Mat::PIXEL_RGBA, w, (in_tile_y1 - in_tile_y0));
            }
        }

        ncnn::VkCompute cmd(vkdev);

        // upload
        ncnn::VkMat in_gpu;
        {
            cmd.record_clone(in, in_gpu, opt);

            if (xtiles > 1)
            {
                cmd.submit_and_wait();
                cmd.reset();
            }
        }

        int out_tile_y0 = std::max(yi * TILE_SIZE_Y, 0);
        int out_tile_y1 = std::min((yi + 1) * TILE_SIZE_Y, h);

        ncnn::VkMat out_gpu;
        if ((opt.use_fp16_storage || opt.use_fp16_packed) && opt.use_int8_storage)
        {
            out_gpu.create(w * scale, (out_tile_y1 - out_tile_y0) * scale, (size_t)channels, 1, opt.blob_vkallocator);
        }
        else
        {
            out_gpu.create(w * scale, (out_tile_y1 - out_tile_y0) * scale, channels, (size_t)4u, 1, opt.blob_vkallocator);
        }

        for (int xi = 0; xi < xtiles; xi++)
        {
            const int tile_w_nopad = std::min((xi + 1) * TILE_SIZE_X, w) - xi * TILE_SIZE_X;

            int prepadding_right = prepadding;
            if (scale == 1 || scale == 3)
            {
                prepadding_right += (tile_w_nopad + 3) / 4 * 4 - tile_w_nopad;
            }
            if (scale == 2 || scale == 4)
            {
                prepadding_right += (tile_w_nopad + 1) / 2 * 2 - tile_w_nopad;
            }

            if (tta_level > 1)
            {
                // preproc
                ncnn::VkMat in_tile_gpu[8];
                ncnn::VkMat in_alpha_tile_gpu;
                ncnn::VkMat dummy_tile_gpu;
                if (tta_level < 8)
                {
                    dummy_tile_gpu.create(1, 1, 3, in_out_tile_elemsize, 1, opt.blob_vkallocator);
                }
                {
                    // crop tile
                    int tile_x0 = xi * TILE_SIZE_X - prepadding;
                    int tile_x1 = std::min((xi + 1) * TILE_SIZE_X, w) + prepadding_right;
                    int tile_y0 = yi * TILE_SIZE_Y - prepadding;
                    int tile_y1 = std::min((yi + 1) * TILE_SIZE_Y, h) + prepadding_bottom;

                    in_tile_gpu[0].create(tile_x1 - tile_x0, tile_y1 - tile_y0, 3, in_out_tile_elemsize, 1, opt.blob_vkallocator);
                    in_tile_gpu[1].create(tile_x1 - tile_x0, tile_y1 - tile_y0, 3, in_out_tile_elemsize, 1, opt.blob_vkallocator);
                    if (tta_level >= 4)
                    {
                        in_tile_gpu[2].create(tile_x1 - tile_x0, tile_y1 - tile_y0, 3, in_out_tile_elemsize, 1, opt.blob_vkallocator);
                        in_tile_gpu[3].create(tile_x1 - tile_x0, tile_y1 - tile_y0, 3, in_out_tile_elemsize, 1, opt.blob_vkallocator);
                    }
                    if (tta_level >= 8)
                    {
                        in_tile_gpu[4].create(tile_y1 - tile_y0, tile_x1 - tile_x0, 3, in_out_tile_elemsize, 1, opt.blob_vkallocator);
                        in_tile_gpu[5].create(tile_y1 - tile_y0, tile_x1 - tile_x0, 3, in_out_tile_elemsize, 1, opt.blob_vkallocator);
                        in_tile_gpu[6].create(tile_y1 - tile_y0, tile_x1 - tile_x0, 3, in_out_tile_elemsize, 1, opt.blob_vkallocator);
                        in_tile_gpu[7].create(tile_y1 - tile_y0, tile_x1 - tile_x0, 3, in_out_tile_elemsize, 1, opt.blob_vkallocator);
                    }

                    if (channels == 4)
                    {
                        in_alpha_tile_gpu.create(tile_w_nopad, tile_h_nopad, 1, in_out_tile_elemsize, 1, opt.blob_vkallocator);
                    }

                    std::vector<ncnn::VkMat> bindings(10);
                    bindings[0] = in_gpu;
                    for (int ti = 0; ti < tta_level; ti++)
                    {
                        bindings[1 + ti] = in_tile_gpu[ti];
                    }
                    for (int ti = tta_level; ti < 8; ti++)
                    {
                        bindings[1 + ti] = dummy_tile_gpu;
                    }
                    bindings[9] = in_alpha_tile_gpu;

                    std::vector<ncnn::vk_constant_type> constants(13);
                    constants[0].i = in_gpu.w;
                    constants[1].i = in_gpu.h;
                    constants[2].i = in_gpu.cstep;
                    constants[3].i = in_tile_gpu[0].w;
                    constants[4].i = in_tile_gpu[0].h;
                    constants[5].i = in_tile_gpu[0].cstep;
                    constants[6].i = prepadding;
                    constants[7].i = prepadding;
                    constants[8].i = xi * TILE_SIZE_X;
                    constants[9].i = std::min(yi * TILE_SIZE_Y, prepadding);
                    constants[10].i = channels;
                    constants[11].i = in_alpha_tile_gpu.w;
                    constants[12].i = in_alpha_tile_gpu.h;

                    ncnn::VkMat dispatcher;
                    dispatcher.w = in_tile_gpu[0].w;
                    dispatcher.h = in_tile_gpu[0].h;
                    dispatcher.c = channels;

                    cmd.record_pipeline(realcugan_preproc, bindings, constants, dispatcher);
                }

                // realcugan
                ncnn::VkMat out_tile_gpu[8];
                for (int ti = 0; ti < tta_level; ti++)
                {
                    ncnn::Extractor ex = net.create_extractor();

                    ex.set_blob_vkallocator(opt.blob_vkallocator);
                    ex.set_workspace_vkallocator(opt.blob_vkallocator);
                    ex.set_staging_vkallocator(opt.staging_vkallocator);

                    ex.input("in0", in_tile_gpu[ti]);

                    for (size_t i = 0; i < names.size(); i++)
                    {
                        ncnn::VkMat feat;
                        cache.load(yi, xi, ti, names[i], feat);

                        ex.input(names[i].c_str(), feat);
                    }

                    int ret = ex.extract("out0", out_tile_gpu[ti], cmd);
                    if (ret != 0) return ret;
                }
                ncnn::VkMat out_alpha_tile_gpu;
                if (channels == 4)
                {
                    if (scale == 1)
                    {
                        out_alpha_tile_gpu = in_alpha_tile_gpu;
                    }
                    if (scale == 2)
                    {
                        bicubic_2x->forward(in_alpha_tile_gpu, out_alpha_tile_gpu, cmd, opt);
                    }
                    if (scale == 3)
                    {
                        bicubic_3x->forward(in_alpha_tile_gpu, out_alpha_tile_gpu, cmd, opt);
                    }
                    if (scale == 4)
                    {
                        bicubic_4x->forward(in_alpha_tile_gpu, out_alpha_tile_gpu, cmd, opt);
                    }
                }

                // postproc
                if (scale == 4)
                {
                    std::vector<ncnn::VkMat> bindings(11);
                    bindings[0] = in_gpu;
                    for (int ti = 0; ti < tta_level; ti++)
                    {
                        bindings[1 + ti] = out_tile_gpu[ti];
                    }
                    for (int ti = tta_level; ti < 8; ti++)
                    {
                        bindings[1 + ti] = dummy_tile_gpu;
                    }
                    bindings[9] = out_alpha_tile_gpu;
                    bindings[10] = out_gpu;

                    std::vector<ncnn::vk_constant_type> constants(16);
                    constants[0].i = in_gpu.w;
                    constants[1].i = in_gpu.h;
                    constants[2].i = in_gpu.cstep;
                    constants[3].i = out_tile_gpu[0].w;
                    constants[4].i = out_tile_gpu[0].h;
                    constants[5].i = out_tile_gpu[0].cstep;
                    constants[6].i = out_gpu.w;
                    constants[7].i = out_gpu.h;
                    constants[8].i = out_gpu.cstep;
                    constants[9].i = xi * TILE_SIZE_X;
                    constants[10].i = std::min(yi * TILE_SIZE_Y, prepadding);
                    constants[11].i = xi * TILE_SIZE_X * scale;
                    constants[12].i = std::min(TILE_SIZE_X * scale, out_gpu.w - xi * TILE_SIZE_X * scale);
                    constants[13].i = channels;
                    constants[14].i = out_alpha_tile_gpu.w;
                    constants[15].i = out_alpha_tile_gpu.h;

                    ncnn::VkMat dispatcher;
                    dispatcher.w = std::min(TILE_SIZE_X * scale, out_gpu.w - xi * TILE_SIZE_X * scale);
                    dispatcher.h = out_gpu.h;
                    dispatcher.c = channels;

                    cmd.record_pipeline(realcugan_4x_postproc, bindings, constants, dispatcher);
                }
                else
                {
                    std::vector<ncnn::VkMat> bindings(10);
                    for (int ti = 0; ti < tta_level; ti++)
                    {
                        bindings[ti] = out_tile_gpu[ti];
                    }
                    for (int ti = tta_level; ti < 8; ti++)
                    {
                        bindings[ti] = dummy_tile_gpu;
                    }
                    bindings[8] = out_alpha_tile_gpu;
                    bindings[9] = out_gpu;

                    std::vector<ncnn::vk_constant_type> constants(11);
                    constants[0].i = out_tile_gpu[0].w;
                    constants[1].i = out_tile_gpu[0].h;
                    constants[2].i = out_tile_gpu[0].cstep;
                    constants[3].i = out_gpu.w;
                    constants[4].i = out_gpu.h;
                    constants[5].i = out_gpu.cstep;
                    constants[6].i = xi * TILE_SIZE_X * scale;
                    constants[7].i = std::min(TILE_SIZE_X * scale, out_gpu.w - xi * TILE_SIZE_X * scale);
                    constants[8].i = channels;
                    constants[9].i = out_alpha_tile_gpu.w;
                    constants[10].i = out_alpha_tile_gpu.h;

                    ncnn::VkMat dispatcher;
                    dispatcher.w = std::min(TILE_SIZE_X * scale, out_gpu.w - xi * TILE_SIZE_X * scale);
                    dispatcher.h = out_gpu.h;
                    dispatcher.c = channels;

                    cmd.record_pipeline(realcugan_postproc, bindings, constants, dispatcher);
                }
            }
            else
            {
                // preproc
                ncnn::VkMat in_tile_gpu;
                ncnn::VkMat in_alpha_tile_gpu;
                {
                    // crop tile
                    int tile_x0 = xi * TILE_SIZE_X - prepadding;
                    int tile_x1 = std::min((xi + 1) * TILE_SIZE_X, w) + prepadding_right;
                    int tile_y0 = yi * TILE_SIZE_Y - prepadding;
                    int tile_y1 = std::min((yi + 1) * TILE_SIZE_Y, h) + prepadding_bottom;

                    in_tile_gpu.create(tile_x1 - tile_x0, tile_y1 - tile_y0, 3, in_out_tile_elemsize, 1, opt.blob_vkallocator);

                    if (channels == 4)
                    {
                        in_alpha_tile_gpu.create(tile_w_nopad, tile_h_nopad, 1, in_out_tile_elemsize, 1, opt.blob_vkallocator);
                    }

                    std::vector<ncnn::VkMat> bindings(3);
                    bindings[0] = in_gpu;
                    bindings[1] = in_tile_gpu;
                    bindings[2] = in_alpha_tile_gpu;

                    std::vector<ncnn::vk_constant_type> constants(13);
                    constants[0].i = in_gpu.w;
                    constants[1].i = in_gpu.h;
                    constants[2].i = in_gpu.cstep;
                    constants[3].i = in_tile_gpu.w;
                    constants[4].i = in_tile_gpu.h;
                    constants[5].i = in_tile_gpu.cstep;
                    constants[6].i = prepadding;
                    constants[7].i = prepadding;
                    constants[8].i = xi * TILE_SIZE_X;
                    constants[9].i = std::min(yi * TILE_SIZE_Y, prepadding);
                    constants[10].i = channels;
                    constants[11].i = in_alpha_tile_gpu.w;
                    constants[12].i = in_alpha_tile_gpu.h;

                    ncnn::VkMat dispatcher;
                    dispatcher.w = in_tile_gpu.w;
                    dispatcher.h = in_tile_gpu.h;
                    dispatcher.c = channels;

                    cmd.record_pipeline(realcugan_preproc, bindings, constants, dispatcher);
                }

                // realcugan
                ncnn::VkMat out_tile_gpu;
                {
                    ncnn::Extractor ex = net.create_extractor();

                    ex.set_blob_vkallocator(opt.blob_vkallocator);
                    ex.set_workspace_vkallocator(opt.blob_vkallocator);
                    ex.set_staging_vkallocator(opt.staging_vkallocator);

                    ex.input("in0", in_tile_gpu);

                    for (size_t i = 0; i < names.size(); i++)
                    {
                        ncnn::VkMat feat;
                        cache.load(yi, xi, 0, names[i], feat);

                        ex.input(names[i].c_str(), feat);
                    }

                    int ret = ex.extract("out0", out_tile_gpu, cmd);
                    if (ret != 0) return ret;
                }

                ncnn::VkMat out_alpha_tile_gpu;
                if (channels == 4)
                {
                    if (scale == 1)
                    {
                        out_alpha_tile_gpu = in_alpha_tile_gpu;
                    }
                    if (scale == 2)
                    {
                        bicubic_2x->forward(in_alpha_tile_gpu, out_alpha_tile_gpu, cmd, opt);
                    }
                    if (scale == 3)
                    {
                        bicubic_3x->forward(in_alpha_tile_gpu, out_alpha_tile_gpu, cmd, opt);
                    }
                    if (scale == 4)
                    {
                        bicubic_4x->forward(in_alpha_tile_gpu, out_alpha_tile_gpu, cmd, opt);
                    }
                }

                // postproc
                if (scale == 4)
                {
                    std::vector<ncnn::VkMat> bindings(4);
                    bindings[0] = in_gpu;
                    bindings[1] = out_tile_gpu;
                    bindings[2] = out_alpha_tile_gpu;
                    bindings[3] = out_gpu;

                    std::vector<ncnn::vk_constant_type> constants(16);
                    constants[0].i = in_gpu.w;
                    constants[1].i = in_gpu.h;
                    constants[2].i = in_gpu.cstep;
                    constants[3].i = out_tile_gpu.w;
                    constants[4].i = out_tile_gpu.h;
                    constants[5].i = out_tile_gpu.cstep;
                    constants[6].i = out_gpu.w;
                    constants[7].i = out_gpu.h;
                    constants[8].i = out_gpu.cstep;
                    constants[9].i = xi * TILE_SIZE_X;
                    constants[10].i = std::min(yi * TILE_SIZE_Y, prepadding);
                    constants[11].i = xi * TILE_SIZE_X * scale;
                    constants[12].i = std::min(TILE_SIZE_X * scale, out_gpu.w - xi * TILE_SIZE_X * scale);
                    constants[13].i = channels;
                    constants[14].i = out_alpha_tile_gpu.w;
                    constants[15].i = out_alpha_tile_gpu.h;

                    ncnn::VkMat dispatcher;
                    dispatcher.w = std::min(TILE_SIZE_X * scale, out_gpu.w - xi * TILE_SIZE_X * scale);
                    dispatcher.h = out_gpu.h;
                    dispatcher.c = channels;

                    cmd.record_pipeline(realcugan_4x_postproc, bindings, constants, dispatcher);
                }
                else
                {
                    std::vector<ncnn::VkMat> bindings(3);
                    bindings[0] = out_tile_gpu;
                    bindings[1] = out_alpha_tile_gpu;
                    bindings[2] = out_gpu;

                    std::vector<ncnn::vk_constant_type> constants(11);
                    constants[0].i = out_tile_gpu.w;
                    constants[1].i = out_tile_gpu.h;
                    constants[2].i = out_tile_gpu.cstep;
                    constants[3].i = out_gpu.w;
                    constants[4].i = out_gpu.h;
                    constants[5].i = out_gpu.cstep;
                    constants[6].i = xi * TILE_SIZE_X * scale;
                    constants[7].i = std::min(TILE_SIZE_X * scale, out_gpu.w - xi * TILE_SIZE_X * scale);
                    constants[8].i = channels;
                    constants[9].i = out_alpha_tile_gpu.w;
                    constants[10].i = out_alpha_tile_gpu.h;

                    ncnn::VkMat dispatcher;
                    dispatcher.w = std::min(TILE_SIZE_X * scale, out_gpu.w - xi * TILE_SIZE_X * scale);
                    dispatcher.h = out_gpu.h;
                    dispatcher.c = channels;

                    cmd.record_pipeline(realcugan_postproc, bindings, constants, dispatcher);
                }
            }

            if (xtiles > 1)
            {
                cmd.submit_and_wait();
                cmd.reset();
            }
        }

        // download
        {
            ncnn::Mat out;

            if ((opt.use_fp16_storage || opt.use_fp16_packed) && opt.use_int8_storage)
            {
                out = ncnn::Mat(out_gpu.w, out_gpu.h, (unsigned char*)outimage.data + (size_t)yi * scale * TILE_SIZE_Y * w * scale * channels, (size_t)channels, 1);
            }

            cmd.record_clone(out_gpu, out, opt);

            cmd.submit_and_wait();

            if (!((opt.use_fp16_storage || opt.use_fp16_packed) && opt.use_int8_storage))
            {
                if (channels == 3)
                {
                    out.to_pixels((unsigned char*)outimage.data + (size_t)yi * scale * TILE_SIZE_Y * w * scale * channels, bgr_mode ? ncnn::Mat::PIXEL_RGB2BGR : ncnn::Mat::PIXEL_RGB);
                }
                if (channels == 4)
                {
                    out.to_pixels((unsigned char*)outimage.data + (size_t)yi * scale * TILE_SIZE_Y * w * scale * channels, bgr_mode ? ncnn::Mat::PIXEL_RGBA2BGRA : ncnn::Mat::PIXEL_RGBA);
                }
            }
        }
    }

    return 0;
}

int RealCUGAN::process_se_sync_gap(const ncnn::Mat& inimage, const std::vector<std::string>& names, const ncnn::Option& opt, FeatureCache& cache, int current_tilesize) const
{
    const unsigned char* pixeldata = (const unsigned char*)inimage.data;
    const int w = inimage.w;
    const int h = inimage.h;
    const int channels = inimage.elempack;

    const int TILE_SIZE_X = current_tilesize > 0 ? current_tilesize : tilesize;
    const int TILE_SIZE_Y = current_tilesize > 0 ? current_tilesize : tilesize;

    // each tile 400x400
    const int xtiles = (w + TILE_SIZE_X - 1) / TILE_SIZE_X;
    const int ytiles = (h + TILE_SIZE_Y - 1) / TILE_SIZE_Y;

    const int tta_count = (tta_level > 1) ? tta_level : 1;
    const int num_ti = (syncgap >= 2) ? 1 : tta_count;

    std::vector< std::vector< std::vector<ncnn::VkMat> > > feats(names.size(), std::vector< std::vector<ncnn::VkMat> >(num_ti));
    for (int yi = 0; yi < ytiles; yi++)
    {
        for (int xi = 0; xi < xtiles; xi++)
        {
            for (size_t i = 0; i < names.size(); i++)
            {
                for (int ti = 0; ti < num_ti; ti++)
                {
                    ncnn::VkMat feat;
                    cache.load(yi, xi, ti, names[i], feat);
                    feats[i][ti].push_back(feat);
                }
            }
        }
    }

    const int tiles = ytiles * xtiles;

    std::vector<float> tile_weights;
    tile_weights.reserve(tiles);
    float total_weight = 0.f;
    for (int yi = 0; yi < ytiles; yi++)
    {
        const int tile_h_nopad = std::min((yi + 1) * TILE_SIZE_Y, h) - yi * TILE_SIZE_Y;
        for (int xi = 0; xi < xtiles; xi++)
        {
            const int tile_w_nopad = std::min((xi + 1) * TILE_SIZE_X, w) - xi * TILE_SIZE_X;
            const float area = (float)(tile_w_nopad * tile_h_nopad);
            tile_weights.push_back(area);
            total_weight += area;
        }
    }

    ncnn::VkCompute cmd(vkdev);

    // download
    std::vector< std::vector< std::vector<ncnn::Mat> > > feats_cpu(names.size(), std::vector< std::vector<ncnn::Mat> >(num_ti));
    for (size_t i = 0; i < names.size(); i++)
    {
        for (int ti = 0; ti < num_ti; ti++)
        {
            feats_cpu[i][ti].resize(tiles);

            for (int j = 0; j < tiles; j++)
            {
                cmd.record_download(feats[i][ti][j], feats_cpu[i][ti][j], opt);
            }
        }
    }

    cmd.submit_and_wait();
    cmd.reset();

    // global average
    // upload
    std::vector< std::vector<ncnn::VkMat> > avgfeats(names.size(), std::vector<ncnn::VkMat>(num_ti));
    for (size_t i = 0; i < names.size(); i++)
    {
        for (int ti = 0; ti < num_ti; ti++)
        {
            for (int j = 0; j < tiles; j++)
            {
                if (feats_cpu[i][ti][j].elembits() == 16)
                {
                    ncnn::Mat feat_fp32;
                    ncnn::cast_float16_to_float32(feats_cpu[i][ti][j], feat_fp32, opt);
                    feats_cpu[i][ti][j] = feat_fp32;
                }

                if (opt.use_packing_layout && feats_cpu[i][ti][j].elempack != 1)
                {
                    ncnn::Mat feat_cpu_unpacked;
                    ncnn::convert_packing(feats_cpu[i][ti][j], feat_cpu_unpacked, 1, opt);
                    feats_cpu[i][ti][j] = feat_cpu_unpacked;
                }
            }

            // handle feats_cpu[i][ti] vector
            {
                ncnn::Mat avgfeat;
                avgfeat.create_like(feats_cpu[i][ti][0]);
                avgfeat.fill(0.f);

                int len = avgfeat.total();

                for (int j = 0; j < tiles; j++)
                {
                    const ncnn::Mat f = feats_cpu[i][ti][j];
                    const float w_j = tile_weights[j];

                    for (int k = 0; k < len; k++)
                    {
                        avgfeat[k] += f[k] * w_j;
                    }
                }

                if (total_weight > 0.f)
                {
                    for (int k = 0; k < len; k++)
                    {
                        avgfeat[k] /= total_weight;
                    }
                }
                else if (tiles > 0)
                {
                    for (int k = 0; k < len; k++)
                    {
                        avgfeat[k] /= tiles;
                    }
                }

                cmd.record_upload(avgfeat, avgfeats[i][ti], opt);
            }
        }
    }

    cmd.submit_and_wait();
    cmd.reset();

    for (int yi = 0; yi < ytiles; yi++)
    {
        for (int xi = 0; xi < xtiles; xi++)
        {
            for (size_t i = 0; i < names.size(); i++)
            {
                for (int ti = 0; ti < tta_count; ti++)
                {
                    int src_ti = (syncgap >= 2) ? 0 : ti;
                    cache.save(yi, xi, ti, names[i], avgfeats[i][src_ti]);
                }
            }
        }
    }

    return 0;
}

int RealCUGAN::process_se_very_rough_stage0(const ncnn::Mat& inimage, const std::vector<std::string>& names, const std::vector<std::string>& outnames, const ncnn::Option& opt, FeatureCache& cache, int current_tilesize) const
{
    const unsigned char* pixeldata = (const unsigned char*)inimage.data;
    const int w = inimage.w;
    const int h = inimage.h;
    const int channels = inimage.elempack;

    const int TILE_SIZE_X = current_tilesize > 0 ? current_tilesize : tilesize;
    const int TILE_SIZE_Y = current_tilesize > 0 ? current_tilesize : tilesize;

    // each tile 400x400
    const int xtiles = (w + TILE_SIZE_X - 1) / TILE_SIZE_X;
    const int ytiles = (h + TILE_SIZE_Y - 1) / TILE_SIZE_Y;

    const size_t in_out_tile_elemsize = (opt.use_fp16_storage || opt.use_fp16_packed) ? 2u : 4u;

    int yi_start = (ytiles >= 3) ? 1 : 0;
    int xi_start = (xtiles >= 3) ? 1 : 0;

    //#pragma omp parallel for num_threads(2)
    for (int yi = yi_start; yi < ytiles; yi += 2)
    {
        const int tile_h_nopad = std::min((yi + 1) * TILE_SIZE_Y, h) - yi * TILE_SIZE_Y;

        int prepadding_bottom = prepadding;
        if (scale == 1 || scale == 3)
        {
            prepadding_bottom += (tile_h_nopad + 3) / 4 * 4 - tile_h_nopad;
        }
        if (scale == 2 || scale == 4)
        {
            prepadding_bottom += (tile_h_nopad + 1) / 2 * 2 - tile_h_nopad;
        }

        int in_tile_y0 = std::max(yi * TILE_SIZE_Y - prepadding, 0);
        int in_tile_y1 = std::min((yi + 1) * TILE_SIZE_Y + prepadding_bottom, h);

        ncnn::Mat in;
        if ((opt.use_fp16_storage || opt.use_fp16_packed) && opt.use_int8_storage)
        {
            in = ncnn::Mat(w, (in_tile_y1 - in_tile_y0), (unsigned char*)pixeldata + (size_t)in_tile_y0 * w * channels, (size_t)channels, 1);
        }
        else
        {
            if (channels == 3)
            {
                in = ncnn::Mat::from_pixels(pixeldata + (size_t)in_tile_y0 * w * channels, bgr_mode ? ncnn::Mat::PIXEL_BGR2RGB : ncnn::Mat::PIXEL_RGB, w, (in_tile_y1 - in_tile_y0));
            }
            if (channels == 4)
            {
                in = ncnn::Mat::from_pixels(pixeldata + (size_t)in_tile_y0 * w * channels, bgr_mode ? ncnn::Mat::PIXEL_BGRA2RGBA : ncnn::Mat::PIXEL_RGBA, w, (in_tile_y1 - in_tile_y0));
            }
        }

        ncnn::VkCompute cmd(vkdev);

        // upload
        ncnn::VkMat in_gpu;
        {
            cmd.record_clone(in, in_gpu, opt);

            if (xtiles > 1)
            {
                cmd.submit_and_wait();
                cmd.reset();
            }
        }

        for (int xi = xi_start; xi < xtiles; xi += 2)
        {
            const int tile_w_nopad = std::min((xi + 1) * TILE_SIZE_X, w) - xi * TILE_SIZE_X;

            int prepadding_right = prepadding;
            if (scale == 1 || scale == 3)
            {
                prepadding_right += (tile_w_nopad + 3) / 4 * 4 - tile_w_nopad;
            }
            if (scale == 2 || scale == 4)
            {
                prepadding_right += (tile_w_nopad + 1) / 2 * 2 - tile_w_nopad;
            }

            if (tta_level > 1)
            {
                // preproc
                ncnn::VkMat in_tile_gpu[8];
                ncnn::VkMat in_alpha_tile_gpu;
                ncnn::VkMat dummy_tile_gpu;
                if (tta_level < 8)
                {
                    dummy_tile_gpu.create(1, 1, 3, in_out_tile_elemsize, 1, opt.blob_vkallocator);
                }
                {
                    // crop tile
                    int tile_x0 = xi * TILE_SIZE_X - prepadding;
                    int tile_x1 = std::min((xi + 1) * TILE_SIZE_X, w) + prepadding_right;
                    int tile_y0 = yi * TILE_SIZE_Y - prepadding;
                    int tile_y1 = std::min((yi + 1) * TILE_SIZE_Y, h) + prepadding_bottom;

                    in_tile_gpu[0].create(tile_x1 - tile_x0, tile_y1 - tile_y0, 3, in_out_tile_elemsize, 1, opt.blob_vkallocator);
                    in_tile_gpu[1].create(tile_x1 - tile_x0, tile_y1 - tile_y0, 3, in_out_tile_elemsize, 1, opt.blob_vkallocator);
                    if (tta_level >= 4)
                    {
                        in_tile_gpu[2].create(tile_x1 - tile_x0, tile_y1 - tile_y0, 3, in_out_tile_elemsize, 1, opt.blob_vkallocator);
                        in_tile_gpu[3].create(tile_x1 - tile_x0, tile_y1 - tile_y0, 3, in_out_tile_elemsize, 1, opt.blob_vkallocator);
                    }
                    if (tta_level >= 8)
                    {
                        in_tile_gpu[4].create(tile_y1 - tile_y0, tile_x1 - tile_x0, 3, in_out_tile_elemsize, 1, opt.blob_vkallocator);
                        in_tile_gpu[5].create(tile_y1 - tile_y0, tile_x1 - tile_x0, 3, in_out_tile_elemsize, 1, opt.blob_vkallocator);
                        in_tile_gpu[6].create(tile_y1 - tile_y0, tile_x1 - tile_x0, 3, in_out_tile_elemsize, 1, opt.blob_vkallocator);
                        in_tile_gpu[7].create(tile_y1 - tile_y0, tile_x1 - tile_x0, 3, in_out_tile_elemsize, 1, opt.blob_vkallocator);
                    }

                    if (channels == 4)
                    {
                        in_alpha_tile_gpu.create(tile_w_nopad, tile_h_nopad, 1, in_out_tile_elemsize, 1, opt.blob_vkallocator);
                    }

                    std::vector<ncnn::VkMat> bindings(10);
                    bindings[0] = in_gpu;
                    for (int ti = 0; ti < tta_level; ti++)
                    {
                        bindings[1 + ti] = in_tile_gpu[ti];
                    }
                    for (int ti = tta_level; ti < 8; ti++)
                    {
                        bindings[1 + ti] = dummy_tile_gpu;
                    }
                    bindings[9] = in_alpha_tile_gpu;

                    std::vector<ncnn::vk_constant_type> constants(13);
                    constants[0].i = in_gpu.w;
                    constants[1].i = in_gpu.h;
                    constants[2].i = in_gpu.cstep;
                    constants[3].i = in_tile_gpu[0].w;
                    constants[4].i = in_tile_gpu[0].h;
                    constants[5].i = in_tile_gpu[0].cstep;
                    constants[6].i = prepadding;
                    constants[7].i = prepadding;
                    constants[8].i = xi * TILE_SIZE_X;
                    constants[9].i = std::min(yi * TILE_SIZE_Y, prepadding);
                    constants[10].i = channels;
                    constants[11].i = in_alpha_tile_gpu.w;
                    constants[12].i = in_alpha_tile_gpu.h;

                    ncnn::VkMat dispatcher;
                    dispatcher.w = in_tile_gpu[0].w;
                    dispatcher.h = in_tile_gpu[0].h;
                    dispatcher.c = channels;

                    cmd.record_pipeline(realcugan_preproc, bindings, constants, dispatcher);
                }

                // realcugan
                int tta_count = (syncgap >= 2) ? 1 : tta_level;
                for (int ti = 0; ti < tta_count; ti++)
                {
                    ncnn::Extractor ex = net.create_extractor();

                    ex.set_blob_vkallocator(opt.blob_vkallocator);
                    ex.set_workspace_vkallocator(opt.blob_vkallocator);
                    ex.set_staging_vkallocator(opt.staging_vkallocator);

                    ex.input("in0", in_tile_gpu[ti]);

                    for (size_t i = 0; i < names.size(); i++)
                    {
                        ncnn::VkMat feat;
                        cache.load(yi, xi, ti, names[i], feat);

                        ex.input(names[i].c_str(), feat);
                    }

                    for (size_t i = 0; i < outnames.size(); i++)
                    {
                        ncnn::VkMat feat;
                        int ret = ex.extract(outnames[i].c_str(), feat, cmd);
                        if (ret != 0) return ret;

                        cache.save(yi, xi, ti, outnames[i], feat);
                    }
                }
            }
            else
            {
                // preproc
                ncnn::VkMat in_tile_gpu;
                ncnn::VkMat in_alpha_tile_gpu;
                {
                    // crop tile
                    int tile_x0 = xi * TILE_SIZE_X - prepadding;
                    int tile_x1 = std::min((xi + 1) * TILE_SIZE_X, w) + prepadding_right;
                    int tile_y0 = yi * TILE_SIZE_Y - prepadding;
                    int tile_y1 = std::min((yi + 1) * TILE_SIZE_Y, h) + prepadding_bottom;

                    in_tile_gpu.create(tile_x1 - tile_x0, tile_y1 - tile_y0, 3, in_out_tile_elemsize, 1, opt.blob_vkallocator);

                    if (channels == 4)
                    {
                        in_alpha_tile_gpu.create(tile_w_nopad, tile_h_nopad, 1, in_out_tile_elemsize, 1, opt.blob_vkallocator);
                    }

                    std::vector<ncnn::VkMat> bindings(3);
                    bindings[0] = in_gpu;
                    bindings[1] = in_tile_gpu;
                    bindings[2] = in_alpha_tile_gpu;

                    std::vector<ncnn::vk_constant_type> constants(13);
                    constants[0].i = in_gpu.w;
                    constants[1].i = in_gpu.h;
                    constants[2].i = in_gpu.cstep;
                    constants[3].i = in_tile_gpu.w;
                    constants[4].i = in_tile_gpu.h;
                    constants[5].i = in_tile_gpu.cstep;
                    constants[6].i = prepadding;
                    constants[7].i = prepadding;
                    constants[8].i = xi * TILE_SIZE_X;
                    constants[9].i = std::min(yi * TILE_SIZE_Y, prepadding);
                    constants[10].i = channels;
                    constants[11].i = in_alpha_tile_gpu.w;
                    constants[12].i = in_alpha_tile_gpu.h;

                    ncnn::VkMat dispatcher;
                    dispatcher.w = in_tile_gpu.w;
                    dispatcher.h = in_tile_gpu.h;
                    dispatcher.c = channels;

                    cmd.record_pipeline(realcugan_preproc, bindings, constants, dispatcher);
                }

                // realcugan
                {
                    ncnn::Extractor ex = net.create_extractor();

                    ex.set_blob_vkallocator(opt.blob_vkallocator);
                    ex.set_workspace_vkallocator(opt.blob_vkallocator);
                    ex.set_staging_vkallocator(opt.staging_vkallocator);

                    ex.input("in0", in_tile_gpu);

                    for (size_t i = 0; i < names.size(); i++)
                    {
                        ncnn::VkMat feat;
                        cache.load(yi, xi, 0, names[i], feat);

                        ex.input(names[i].c_str(), feat);
                    }

                    for (size_t i = 0; i < outnames.size(); i++)
                    {
                        ncnn::VkMat feat;
                        int ret = ex.extract(outnames[i].c_str(), feat, cmd);
                        if (ret != 0) return ret;

                        cache.save(yi, xi, 0, outnames[i], feat);
                    }
                }
            }

            if (xtiles > 1)
            {
                cmd.submit_and_wait();
                cmd.reset();
            }
        }

        if (xtiles <= 1)
        {
            cmd.submit_and_wait();
            cmd.reset();
        }
    }

    return 0;
}

int RealCUGAN::process_se_very_rough_sync_gap(const ncnn::Mat& inimage, const std::vector<std::string>& names, const ncnn::Option& opt, FeatureCache& cache, int current_tilesize) const
{
    const unsigned char* pixeldata = (const unsigned char*)inimage.data;
    const int w = inimage.w;
    const int h = inimage.h;
    const int channels = inimage.elempack;

    const int TILE_SIZE_X = current_tilesize > 0 ? current_tilesize : tilesize;
    const int TILE_SIZE_Y = current_tilesize > 0 ? current_tilesize : tilesize;

    // each tile 400x400
    const int xtiles = (w + TILE_SIZE_X - 1) / TILE_SIZE_X;
    const int ytiles = (h + TILE_SIZE_Y - 1) / TILE_SIZE_Y;

    std::vector< std::vector<ncnn::VkMat> > feats(names.size());
    std::vector<float> tile_weights;
    float total_weight = 0.f;

    int yi_start = (ytiles >= 3) ? 1 : 0;
    int xi_start = (xtiles >= 3) ? 1 : 0;

    for (int yi = yi_start; yi < ytiles; yi += 2)
    {
        const int tile_h_nopad = std::min((yi + 1) * TILE_SIZE_Y, h) - yi * TILE_SIZE_Y;
        for (int xi = xi_start; xi < xtiles; xi += 2)
        {
            const int tile_w_nopad = std::min((xi + 1) * TILE_SIZE_X, w) - xi * TILE_SIZE_X;
            const float area = (float)(tile_w_nopad * tile_h_nopad);
            tile_weights.push_back(area);
            total_weight += area;

            for (size_t i = 0; i < names.size(); i++)
            {
                ncnn::VkMat feat;
                cache.load(yi, xi, 0, names[i], feat);
                feats[i].push_back(feat);
            }
        }
    }

    const int tiles = (int)tile_weights.size();

    ncnn::VkCompute cmd(vkdev);

    // download
    std::vector< std::vector<ncnn::Mat> > feats_cpu(names.size());
    for (size_t i = 0; i < names.size(); i++)
    {
        feats_cpu[i].resize(tiles);

        for (int j = 0; j < tiles; j++)
        {
            cmd.record_download(feats[i][j], feats_cpu[i][j], opt);
        }
    }

    cmd.submit_and_wait();
    cmd.reset();

    // global average
    // upload
    std::vector<ncnn::VkMat> avgfeats(names.size());
    for (size_t i = 0; i < names.size(); i++)
    {
        for (int j = 0; j < tiles; j++)
        {
            if (feats_cpu[i][j].elembits() == 16)
            {
                ncnn::Mat feat_fp32;
                ncnn::cast_float16_to_float32(feats_cpu[i][j], feat_fp32, opt);
                feats_cpu[i][j] = feat_fp32;
            }

            if (opt.use_packing_layout && feats_cpu[i][j].elempack != 1)
            {
                ncnn::Mat feat_cpu_unpacked;
                ncnn::convert_packing(feats_cpu[i][j], feat_cpu_unpacked, 1, opt);
                feats_cpu[i][j] = feat_cpu_unpacked;
            }
        }

        // handle feats_cpu[i] vector
        {
            ncnn::Mat avgfeat;
            avgfeat.create_like(feats_cpu[i][0]);
            avgfeat.fill(0.f);

            int len = avgfeat.total();

            for (int j = 0; j < tiles; j++)
            {
                const ncnn::Mat f = feats_cpu[i][j];
                const float w_j = tile_weights[j];

                for (int k = 0; k < len; k++)
                {
                    avgfeat[k] += f[k] * w_j;
                }
            }

            if (total_weight > 0.f)
            {
                for (int k = 0; k < len; k++)
                {
                    avgfeat[k] /= total_weight;
                }
            }
            else if (tiles > 0)
            {
                for (int k = 0; k < len; k++)
                {
                    avgfeat[k] /= tiles;
                }
            }

            cmd.record_upload(avgfeat, avgfeats[i], opt);
        }
    }

    cmd.submit_and_wait();
    cmd.reset();

    const int tta_count = (tta_level > 1) ? tta_level : 1;
    for (int yi = 0; yi < ytiles; yi++)
    {
        for (int xi = 0; xi < xtiles; xi++)
        {
            for (size_t i = 0; i < names.size(); i++)
            {
                for (int ti = 0; ti < tta_count; ti++)
                {
                    cache.save(yi, xi, ti, names[i], avgfeats[i]);
                }
            }
        }
    }

    return 0;
}

int RealCUGAN::process_cpu_se_stage0(const ncnn::Mat& inimage, const std::vector<std::string>& names, const std::vector<std::string>& outnames, FeatureCache& cache, int current_tilesize) const
{
    const unsigned char* pixeldata = (const unsigned char*)inimage.data;
    const int w = inimage.w;
    const int h = inimage.h;
    const int channels = inimage.elempack;

    const int TILE_SIZE_X = current_tilesize > 0 ? current_tilesize : tilesize;
    const int TILE_SIZE_Y = current_tilesize > 0 ? current_tilesize : tilesize;

    ncnn::Option opt = net.opt;

    // each tile 400x400
    const int xtiles = (w + TILE_SIZE_X - 1) / TILE_SIZE_X;
    const int ytiles = (h + TILE_SIZE_Y - 1) / TILE_SIZE_Y;

    for (int yi = 0; yi < ytiles; yi++)
    {
        const int tile_h_nopad = std::min((yi + 1) * TILE_SIZE_Y, h) - yi * TILE_SIZE_Y;

        int prepadding_bottom = prepadding;
        if (scale == 1 || scale == 3)
        {
            prepadding_bottom += (tile_h_nopad + 3) / 4 * 4 - tile_h_nopad;
        }
        if (scale == 2 || scale == 4)
        {
            prepadding_bottom += (tile_h_nopad + 1) / 2 * 2 - tile_h_nopad;
        }

        int in_tile_y0 = std::max(yi * TILE_SIZE_Y - prepadding, 0);
        int in_tile_y1 = std::min((yi + 1) * TILE_SIZE_Y + prepadding_bottom, h);

        for (int xi = 0; xi < xtiles; xi++)
        {
            const int tile_w_nopad = std::min((xi + 1) * TILE_SIZE_X, w) - xi * TILE_SIZE_X;

            int prepadding_right = prepadding;
            if (scale == 1 || scale == 3)
            {
                prepadding_right += (tile_w_nopad + 3) / 4 * 4 - tile_w_nopad;
            }
            if (scale == 2 || scale == 4)
            {
                prepadding_right += (tile_w_nopad + 1) / 2 * 2 - tile_w_nopad;
            }

            int in_tile_x0 = std::max(xi * TILE_SIZE_X - prepadding, 0);
            int in_tile_x1 = std::min((xi + 1) * TILE_SIZE_X + prepadding_right, w);

            // crop tile
            ncnn::Mat in, in_nopad;
            {
                if (channels == 3)
                {
                    in = ncnn::Mat::from_pixels_roi(pixeldata, bgr_mode ? ncnn::Mat::PIXEL_BGR2RGB : ncnn::Mat::PIXEL_RGB, w, h, in_tile_x0, in_tile_y0, in_tile_x1 - in_tile_x0, in_tile_y1 - in_tile_y0);
                }
                if (channels == 4)
                {
                    in_nopad = ncnn::Mat::from_pixels_roi(pixeldata, bgr_mode ? ncnn::Mat::PIXEL_BGRA2RGBA : ncnn::Mat::PIXEL_RGBA, w, h, xi * TILE_SIZE_X, yi * TILE_SIZE_Y, tile_w_nopad, tile_h_nopad);
                    in = ncnn::Mat::from_pixels_roi(pixeldata, bgr_mode ? ncnn::Mat::PIXEL_BGRA2RGBA : ncnn::Mat::PIXEL_RGBA, w, h, in_tile_x0, in_tile_y0, in_tile_x1 - in_tile_x0, in_tile_y1 - in_tile_y0);
                }
            }

            ncnn::Mat out;

            if (tta_level > 1)
            {
                // split alpha and preproc
                ncnn::Mat in_tile[8];
                ncnn::Mat in_alpha_tile;
                {
                    in_tile[0].create(in.w, in.h, 3);
                    for (int q = 0; q < 3; q++)
                    {
                        const float* ptr = in.channel(q);
                        float* outptr0 = in_tile[0].channel(q);

                        const float norm_scale = is_pro_model ? (0.7f / 255.f) : (1.f / 255.f);
                        const float norm_bias = is_pro_model ? 0.15f : 0.0f;

                        for (int i = 0; i < in.h; i++)
                        {
                            for (int j = 0; j < in.w; j++)
                            {
                                *outptr0++ = *ptr++ * norm_scale + norm_bias;
                            }
                        }
                    }

                    if (channels == 4)
                    {
                        in_alpha_tile = in_nopad.channel_range(3, 1).clone();
                    }
                }

                // border padding
                {
                    int pad_top = std::max(prepadding - yi * TILE_SIZE_Y, 0);
                    int pad_bottom = std::max(std::min((yi + 1) * TILE_SIZE_Y + prepadding_bottom - h, prepadding_bottom), 0);
                    int pad_left = std::max(prepadding - xi * TILE_SIZE_X, 0);
                    int pad_right = std::max(std::min((xi + 1) * TILE_SIZE_X + prepadding_right - w, prepadding_right), 0);

                    ncnn::Mat in_tile_padded;
                    ncnn::copy_make_border(in_tile[0], in_tile_padded, pad_top, pad_bottom, pad_left, pad_right, 2, 0.f, net.opt);
                    in_tile[0] = in_tile_padded;
                }

                // the other directions up to tta_level
                in_tile[1].create(in_tile[0].w, in_tile[0].h, 3);
                if (tta_level >= 4)
                {
                    in_tile[2].create(in_tile[0].w, in_tile[0].h, 3);
                    in_tile[3].create(in_tile[0].w, in_tile[0].h, 3);
                }
                if (tta_level >= 8)
                {
                    in_tile[4].create(in_tile[0].h, in_tile[0].w, 3);
                    in_tile[5].create(in_tile[0].h, in_tile[0].w, 3);
                    in_tile[6].create(in_tile[0].h, in_tile[0].w, 3);
                    in_tile[7].create(in_tile[0].h, in_tile[0].w, 3);
                }

                for (int q = 0; q < 3; q++)
                {
                    const ncnn::Mat in_tile_0 = in_tile[0].channel(q);
                    ncnn::Mat in_tile_1 = in_tile[1].channel(q);
                    ncnn::Mat in_tile_2 = (tta_level >= 4) ? in_tile[2].channel(q) : ncnn::Mat();
                    ncnn::Mat in_tile_3 = (tta_level >= 4) ? in_tile[3].channel(q) : ncnn::Mat();
                    ncnn::Mat in_tile_4 = (tta_level >= 8) ? in_tile[4].channel(q) : ncnn::Mat();
                    ncnn::Mat in_tile_5 = (tta_level >= 8) ? in_tile[5].channel(q) : ncnn::Mat();
                    ncnn::Mat in_tile_6 = (tta_level >= 8) ? in_tile[6].channel(q) : ncnn::Mat();
                    ncnn::Mat in_tile_7 = (tta_level >= 8) ? in_tile[7].channel(q) : ncnn::Mat();

                    for (int i = 0; i < in_tile[0].h; i++)
                    {
                        const float* outptr0 = in_tile_0.row(i);
                        float* outptr1 = in_tile_1.row(i) + in_tile[0].w - 1;
                        float* outptr2 = (tta_level >= 4) ? (in_tile_2.row(in_tile[0].h - 1 - i) + in_tile[0].w - 1) : nullptr;
                        float* outptr3 = (tta_level >= 4) ? in_tile_3.row(in_tile[0].h - 1 - i) : nullptr;

                        for (int j = 0; j < in_tile[0].w; j++)
                        {
                            float v = *outptr0++;

                            *outptr1-- = v;
                            if (tta_level >= 4)
                            {
                                *outptr2-- = v;
                                *outptr3++ = v;
                            }
                            if (tta_level >= 8)
                            {
                                float* outptr4 = in_tile_4.row(j) + i;
                                float* outptr5 = in_tile_5.row(in_tile[0].w - 1 - j) + i;
                                float* outptr6 = in_tile_6.row(j) + in_tile[0].h - 1 - i;
                                float* outptr7 = in_tile_7.row(in_tile[0].w - 1 - j) + in_tile[0].h - 1 - i;
                                *outptr4 = v;
                                *outptr5 = v;
                                *outptr6 = v;
                                *outptr7 = v;
                            }
                        }
                    }
                }

                // realcugan
                int tta_count = (syncgap >= 2) ? 1 : tta_level;
                for (int ti = 0; ti < tta_count; ti++)
                {
                    ncnn::Extractor ex = net.create_extractor();

                    ex.input("in0", in_tile[ti]);

                    for (size_t i = 0; i < names.size(); i++)
                    {
                        ncnn::Mat feat;
                        cache.load(yi, xi, ti, names[i], feat);

                        ex.input(names[i].c_str(), feat);
                    }

                    for (size_t i = 0; i < outnames.size(); i++)
                    {
                        ncnn::Mat feat;
                        int ret = ex.extract(outnames[i].c_str(), feat);
                        if (ret != 0) return ret;

                        cache.save(yi, xi, ti, outnames[i], feat);
                    }
                }
            }
            else
            {
                // split alpha and preproc
                ncnn::Mat in_tile;
                ncnn::Mat in_alpha_tile;
                {
                    in_tile.create(in.w, in.h, 3);
                    for (int q = 0; q < 3; q++)
                    {
                        const float* ptr = in.channel(q);
                        float* outptr = in_tile.channel(q);

                        const float norm_scale = is_pro_model ? (0.7f / 255.f) : (1.f / 255.f);
                        const float norm_bias = is_pro_model ? 0.15f : 0.0f;

                        for (int i = 0; i < in.w * in.h; i++)
                        {
                            *outptr++ = *ptr++ * norm_scale + norm_bias;
                        }
                    }

                    if (channels == 4)
                    {
                        in_alpha_tile = in_nopad.channel_range(3, 1).clone();
                    }
                }

                // border padding
                {
                    int pad_top = std::max(prepadding - yi * TILE_SIZE_Y, 0);
                    int pad_bottom = std::max(std::min((yi + 1) * TILE_SIZE_Y + prepadding_bottom - h, prepadding_bottom), 0);
                    int pad_left = std::max(prepadding - xi * TILE_SIZE_X, 0);
                    int pad_right = std::max(std::min((xi + 1) * TILE_SIZE_X + prepadding_right - w, prepadding_right), 0);

                    ncnn::Mat in_tile_padded;
                    ncnn::copy_make_border(in_tile, in_tile_padded, pad_top, pad_bottom, pad_left, pad_right, 2, 0.f, net.opt);
                    in_tile = in_tile_padded;
                }

                {
                    ncnn::Extractor ex = net.create_extractor();

                    ex.input("in0", in_tile);

                    for (size_t i = 0; i < names.size(); i++)
                    {
                        ncnn::Mat feat;
                        cache.load(yi, xi, 0, names[i], feat);

                        ex.input(names[i].c_str(), feat);
                    }

                    for (size_t i = 0; i < outnames.size(); i++)
                    {
                        ncnn::Mat feat;
                        int ret = ex.extract(outnames[i].c_str(), feat);
                        if (ret != 0) return ret;

                        cache.save(yi, xi, 0, outnames[i], feat);
                    }
                }
            }
        }
    }

    return 0;
}

int RealCUGAN::process_cpu_se_stage2(const ncnn::Mat& inimage, const std::vector<std::string>& names, ncnn::Mat& outimage, FeatureCache& cache, int current_tilesize) const
{
    const unsigned char* pixeldata = (const unsigned char*)inimage.data;
    const int w = inimage.w;
    const int h = inimage.h;
    const int channels = inimage.elempack;

    const int TILE_SIZE_X = current_tilesize > 0 ? current_tilesize : tilesize;
    const int TILE_SIZE_Y = current_tilesize > 0 ? current_tilesize : tilesize;

    ncnn::Option opt = net.opt;

    // each tile 400x400
    const int xtiles = (w + TILE_SIZE_X - 1) / TILE_SIZE_X;
    const int ytiles = (h + TILE_SIZE_Y - 1) / TILE_SIZE_Y;

    for (int yi = 0; yi < ytiles; yi++)
    {
        const int tile_h_nopad = std::min((yi + 1) * TILE_SIZE_Y, h) - yi * TILE_SIZE_Y;

        int prepadding_bottom = prepadding;
        if (scale == 1 || scale == 3)
        {
            prepadding_bottom += (tile_h_nopad + 3) / 4 * 4 - tile_h_nopad;
        }
        if (scale == 2 || scale == 4)
        {
            prepadding_bottom += (tile_h_nopad + 1) / 2 * 2 - tile_h_nopad;
        }

        int in_tile_y0 = std::max(yi * TILE_SIZE_Y - prepadding, 0);
        int in_tile_y1 = std::min((yi + 1) * TILE_SIZE_Y + prepadding_bottom, h);

        for (int xi = 0; xi < xtiles; xi++)
        {
            const int tile_w_nopad = std::min((xi + 1) * TILE_SIZE_X, w) - xi * TILE_SIZE_X;

            int prepadding_right = prepadding;
            if (scale == 1 || scale == 3)
            {
                prepadding_right += (tile_w_nopad + 3) / 4 * 4 - tile_w_nopad;
            }
            if (scale == 2 || scale == 4)
            {
                prepadding_right += (tile_w_nopad + 1) / 2 * 2 - tile_w_nopad;
            }

            int in_tile_x0 = std::max(xi * TILE_SIZE_X - prepadding, 0);
            int in_tile_x1 = std::min((xi + 1) * TILE_SIZE_X + prepadding_right, w);

            // crop tile
            ncnn::Mat in, in_nopad;
            {
                if (channels == 3)
                {
                    in = ncnn::Mat::from_pixels_roi(pixeldata, bgr_mode ? ncnn::Mat::PIXEL_BGR2RGB : ncnn::Mat::PIXEL_RGB, w, h, in_tile_x0, in_tile_y0, in_tile_x1 - in_tile_x0, in_tile_y1 - in_tile_y0);
                }
                if (channels == 4)
                {
                    in_nopad = ncnn::Mat::from_pixels_roi(pixeldata, bgr_mode ? ncnn::Mat::PIXEL_BGRA2RGBA : ncnn::Mat::PIXEL_RGBA, w, h, xi * TILE_SIZE_X, yi * TILE_SIZE_Y, tile_w_nopad, tile_h_nopad);
                    in = ncnn::Mat::from_pixels_roi(pixeldata, bgr_mode ? ncnn::Mat::PIXEL_BGRA2RGBA : ncnn::Mat::PIXEL_RGBA, w, h, in_tile_x0, in_tile_y0, in_tile_x1 - in_tile_x0, in_tile_y1 - in_tile_y0);
                }
            }

            ncnn::Mat out;

            if (tta_level > 1)
            {
                // split alpha and preproc
                ncnn::Mat in_tile[8];
                ncnn::Mat in_alpha_tile;
                {
                    in_tile[0].create(in.w, in.h, 3);
                    for (int q = 0; q < 3; q++)
                    {
                        const float* ptr = in.channel(q);
                        float* outptr0 = in_tile[0].channel(q);

                        const float norm_scale = is_pro_model ? (0.7f / 255.f) : (1.f / 255.f);
                        const float norm_bias = is_pro_model ? 0.15f : 0.0f;

                        for (int i = 0; i < in.h; i++)
                        {
                            for (int j = 0; j < in.w; j++)
                            {
                                *outptr0++ = *ptr++ * norm_scale + norm_bias;
                            }
                        }
                    }

                    if (channels == 4)
                    {
                        in_alpha_tile = in_nopad.channel_range(3, 1).clone();
                    }
                }

                // border padding
                {
                    int pad_top = std::max(prepadding - yi * TILE_SIZE_Y, 0);
                    int pad_bottom = std::max(std::min((yi + 1) * TILE_SIZE_Y + prepadding_bottom - h, prepadding_bottom), 0);
                    int pad_left = std::max(prepadding - xi * TILE_SIZE_X, 0);
                    int pad_right = std::max(std::min((xi + 1) * TILE_SIZE_X + prepadding_right - w, prepadding_right), 0);

                    ncnn::Mat in_tile_padded;
                    ncnn::copy_make_border(in_tile[0], in_tile_padded, pad_top, pad_bottom, pad_left, pad_right, 2, 0.f, net.opt);
                    in_tile[0] = in_tile_padded;
                }

                // create symmetries up to tta_level
                in_tile[1].create(in_tile[0].w, in_tile[0].h, 3);
                if (tta_level >= 4)
                {
                    in_tile[2].create(in_tile[0].w, in_tile[0].h, 3);
                    in_tile[3].create(in_tile[0].w, in_tile[0].h, 3);
                }
                if (tta_level >= 8)
                {
                    in_tile[4].create(in_tile[0].h, in_tile[0].w, 3);
                    in_tile[5].create(in_tile[0].h, in_tile[0].w, 3);
                    in_tile[6].create(in_tile[0].h, in_tile[0].w, 3);
                    in_tile[7].create(in_tile[0].h, in_tile[0].w, 3);
                }

                for (int q = 0; q < 3; q++)
                {
                    const ncnn::Mat in_tile_0 = in_tile[0].channel(q);
                    ncnn::Mat in_tile_1 = in_tile[1].channel(q);
                    ncnn::Mat in_tile_2 = (tta_level >= 4) ? in_tile[2].channel(q) : ncnn::Mat();
                    ncnn::Mat in_tile_3 = (tta_level >= 4) ? in_tile[3].channel(q) : ncnn::Mat();
                    ncnn::Mat in_tile_4 = (tta_level >= 8) ? in_tile[4].channel(q) : ncnn::Mat();
                    ncnn::Mat in_tile_5 = (tta_level >= 8) ? in_tile[5].channel(q) : ncnn::Mat();
                    ncnn::Mat in_tile_6 = (tta_level >= 8) ? in_tile[6].channel(q) : ncnn::Mat();
                    ncnn::Mat in_tile_7 = (tta_level >= 8) ? in_tile[7].channel(q) : ncnn::Mat();

                    for (int i = 0; i < in_tile[0].h; i++)
                    {
                        const float* outptr0 = in_tile_0.row(i);
                        float* outptr1 = in_tile_1.row(i) + in_tile[0].w - 1;
                        float* outptr2 = (tta_level >= 4) ? (in_tile_2.row(in_tile[0].h - 1 - i) + in_tile[0].w - 1) : nullptr;
                        float* outptr3 = (tta_level >= 4) ? in_tile_3.row(in_tile[0].h - 1 - i) : nullptr;

                        for (int j = 0; j < in_tile[0].w; j++)
                        {
                            float v = *outptr0++;

                            *outptr1-- = v;
                            if (tta_level >= 4)
                            {
                                *outptr2-- = v;
                                *outptr3++ = v;
                            }
                            if (tta_level >= 8)
                            {
                                float* outptr4 = in_tile_4.row(j) + i;
                                float* outptr5 = in_tile_5.row(in_tile[0].w - 1 - j) + i;
                                float* outptr6 = in_tile_6.row(j) + in_tile[0].h - 1 - i;
                                float* outptr7 = in_tile_7.row(in_tile[0].w - 1 - j) + in_tile[0].h - 1 - i;
                                *outptr4 = v;
                                *outptr5 = v;
                                *outptr6 = v;
                                *outptr7 = v;
                            }
                        }
                    }
                }

                // realcugan
                ncnn::Mat out_tile[8];
                for (int ti = 0; ti < tta_level; ti++)
                {
                    ncnn::Extractor ex = net.create_extractor();

                    ex.input("in0", in_tile[ti]);

                    for (size_t i = 0; i < names.size(); i++)
                    {
                        ncnn::Mat feat;
                        cache.load(yi, xi, ti, names[i], feat);

                        ex.input(names[i].c_str(), feat);
                    }

                    int ret = ex.extract("out0", out_tile[ti]);
                    if (ret != 0) return ret;
                }

                ncnn::Mat out_alpha_tile;
                if (channels == 4)
                {
                    if (scale == 1)
                    {
                        out_alpha_tile = in_alpha_tile;
                    }
                    if (scale == 2)
                    {
                        bicubic_2x->forward(in_alpha_tile, out_alpha_tile, opt);
                    }
                    if (scale == 3)
                    {
                        bicubic_3x->forward(in_alpha_tile, out_alpha_tile, opt);
                    }
                    if (scale == 4)
                    {
                        bicubic_4x->forward(in_alpha_tile, out_alpha_tile, opt);
                    }
                }

                // postproc and merge alpha
                {
                    out.create(tile_w_nopad * scale, tile_h_nopad * scale, channels);
                    if (scale == 4)
                    {
                        for (int q = 0; q < 3; q++)
                        {
                            const ncnn::Mat out_tile_0 = out_tile[0].channel(q);
                            const ncnn::Mat out_tile_1 = out_tile[1].channel(q);
                            const ncnn::Mat out_tile_2 = (tta_level >= 4) ? out_tile[2].channel(q) : ncnn::Mat();
                            const ncnn::Mat out_tile_3 = (tta_level >= 4) ? out_tile[3].channel(q) : ncnn::Mat();
                            const ncnn::Mat out_tile_4 = (tta_level >= 8) ? out_tile[4].channel(q) : ncnn::Mat();
                            const ncnn::Mat out_tile_5 = (tta_level >= 8) ? out_tile[5].channel(q) : ncnn::Mat();
                            const ncnn::Mat out_tile_6 = (tta_level >= 8) ? out_tile[6].channel(q) : ncnn::Mat();
                            const ncnn::Mat out_tile_7 = (tta_level >= 8) ? out_tile[7].channel(q) : ncnn::Mat();
                            float* outptr = out.channel(q);

                            for (int i = 0; i < out.h; i++)
                            {
                                const float* inptr = in_tile[0].channel(q).row(prepadding + i / 4) + prepadding;
                                const float* ptr0 = out_tile_0.row(i);
                                const float* ptr1 = out_tile_1.row(i) + out_tile[0].w - 1;
                                const float* ptr2 = (tta_level >= 4) ? (out_tile_2.row(out_tile[0].h - 1 - i) + out_tile[0].w - 1) : nullptr;
                                const float* ptr3 = (tta_level >= 4) ? out_tile_3.row(out_tile[0].h - 1 - i) : nullptr;

                                for (int j = 0; j < out.w; j++)
                                {
                                    float v0 = *ptr0++;
                                    float v1 = *ptr1--;
                                    float v = 0.f;
                                    if (tta_level == 2)
                                    {
                                        v = (v0 + v1) * 0.5f;
                                    }
                                    else if (tta_level == 4)
                                    {
                                        float v2 = *ptr2--;
                                        float v3 = *ptr3++;
                                        v = (v0 + v1 + v2 + v3) * 0.25f;
                                    }
                                    else
                                    {
                                        float v2 = *ptr2--;
                                        float v3 = *ptr3++;
                                        float v4 = *(out_tile_4.row(j) + i);
                                        float v5 = *(out_tile_5.row(out_tile[0].w - 1 - j) + i);
                                        float v6 = *(out_tile_6.row(j) + out_tile[0].h - 1 - i);
                                        float v7 = *(out_tile_7.row(out_tile[0].w - 1 - j) + out_tile[0].h - 1 - i);
                                        v = (v0 + v1 + v2 + v3 + v4 + v5 + v6 + v7) * 0.125f;
                                    }

                                    if (is_pro_model)
                                    {
                                        v = (v + inptr[j / 4] - 0.15f) * (255.f / 0.7f);
                                    }
                                    else
                                    {
                                        v = v * 255.f + 0.5f + inptr[j / 4] * 255.f;
                                    }

                                    *outptr++ = std::max(0.f, std::min(255.f, v));
                                }
                            }
                        }
                    }
                    else
                    {
                        for (int q = 0; q < 3; q++)
                        {
                            const ncnn::Mat out_tile_0 = out_tile[0].channel(q);
                            const ncnn::Mat out_tile_1 = out_tile[1].channel(q);
                            const ncnn::Mat out_tile_2 = (tta_level >= 4) ? out_tile[2].channel(q) : ncnn::Mat();
                            const ncnn::Mat out_tile_3 = (tta_level >= 4) ? out_tile[3].channel(q) : ncnn::Mat();
                            const ncnn::Mat out_tile_4 = (tta_level >= 8) ? out_tile[4].channel(q) : ncnn::Mat();
                            const ncnn::Mat out_tile_5 = (tta_level >= 8) ? out_tile[5].channel(q) : ncnn::Mat();
                            const ncnn::Mat out_tile_6 = (tta_level >= 8) ? out_tile[6].channel(q) : ncnn::Mat();
                            const ncnn::Mat out_tile_7 = (tta_level >= 8) ? out_tile[7].channel(q) : ncnn::Mat();
                            float* outptr = out.channel(q);

                            for (int i = 0; i < out.h; i++)
                            {
                                const float* ptr0 = out_tile_0.row(i);
                                const float* ptr1 = out_tile_1.row(i) + out_tile[0].w - 1;
                                const float* ptr2 = (tta_level >= 4) ? (out_tile_2.row(out_tile[0].h - 1 - i) + out_tile[0].w - 1) : nullptr;
                                const float* ptr3 = (tta_level >= 4) ? out_tile_3.row(out_tile[0].h - 1 - i) : nullptr;

                                for (int j = 0; j < out.w; j++)
                                {
                                    float v0 = *ptr0++;
                                    float v1 = *ptr1--;
                                    float v = 0.f;
                                    if (tta_level == 2)
                                    {
                                        v = (v0 + v1) * 0.5f;
                                    }
                                    else if (tta_level == 4)
                                    {
                                        float v2 = *ptr2--;
                                        float v3 = *ptr3++;
                                        v = (v0 + v1 + v2 + v3) * 0.25f;
                                    }
                                    else
                                    {
                                        float v2 = *ptr2--;
                                        float v3 = *ptr3++;
                                        float v4 = *(out_tile_4.row(j) + i);
                                        float v5 = *(out_tile_5.row(out_tile[0].w - 1 - j) + i);
                                        float v6 = *(out_tile_6.row(j) + out_tile[0].h - 1 - i);
                                        float v7 = *(out_tile_7.row(out_tile[0].w - 1 - j) + out_tile[0].h - 1 - i);
                                        v = (v0 + v1 + v2 + v3 + v4 + v5 + v6 + v7) * 0.125f;
                                    }

                                    if (is_pro_model)
                                        v = (v - 0.15f) * (255.f / 0.7f);
                                    else
                                        v = v * 255.f;

                                    *outptr++ = std::max(0.f, std::min(255.f, v + 0.5f));
                                }
                            }
                        }
                    }

                    if (channels == 4)
                    {
                        memcpy(out.channel_range(3, 1), out_alpha_tile, out_alpha_tile.total() * sizeof(float));
                    }
                }
            }
            else
            {
                // split alpha and preproc
                ncnn::Mat in_tile;
                ncnn::Mat in_alpha_tile;
                {
                    in_tile.create(in.w, in.h, 3);
                    for (int q = 0; q < 3; q++)
                    {
                        const float* ptr = in.channel(q);
                        float* outptr = in_tile.channel(q);

                        const float norm_scale = is_pro_model ? (0.7f / 255.f) : (1.f / 255.f);
                        const float norm_bias = is_pro_model ? 0.15f : 0.0f;

                        for (int i = 0; i < in.w * in.h; i++)
                        {
                            *outptr++ = *ptr++ * norm_scale + norm_bias;
                        }
                    }

                    if (channels == 4)
                    {
                        in_alpha_tile = in_nopad.channel_range(3, 1).clone();
                    }
                }

                // border padding
                {
                    int pad_top = std::max(prepadding - yi * TILE_SIZE_Y, 0);
                    int pad_bottom = std::max(std::min((yi + 1) * TILE_SIZE_Y + prepadding_bottom - h, prepadding_bottom), 0);
                    int pad_left = std::max(prepadding - xi * TILE_SIZE_X, 0);
                    int pad_right = std::max(std::min((xi + 1) * TILE_SIZE_X + prepadding_right - w, prepadding_right), 0);

                    ncnn::Mat in_tile_padded;
                    ncnn::copy_make_border(in_tile, in_tile_padded, pad_top, pad_bottom, pad_left, pad_right, 2, 0.f, net.opt);
                    in_tile = in_tile_padded;
                }

                // realcugan
                ncnn::Mat out_tile;
                {
                    ncnn::Extractor ex = net.create_extractor();

                    ex.input("in0", in_tile);

                    for (size_t i = 0; i < names.size(); i++)
                    {
                        ncnn::Mat feat;
                        cache.load(yi, xi, 0, names[i], feat);

                        ex.input(names[i].c_str(), feat);
                    }

                    int ret = ex.extract("out0", out_tile);
                    if (ret != 0) return ret;
                }

                ncnn::Mat out_alpha_tile;
                if (channels == 4)
                {
                    if (scale == 1)
                    {
                        out_alpha_tile = in_alpha_tile;
                    }
                    if (scale == 2)
                    {
                        bicubic_2x->forward(in_alpha_tile, out_alpha_tile, opt);
                    }
                    if (scale == 3)
                    {
                        bicubic_3x->forward(in_alpha_tile, out_alpha_tile, opt);
                    }
                    if (scale == 4)
                    {
                        bicubic_4x->forward(in_alpha_tile, out_alpha_tile, opt);
                    }
                }

                // postproc and merge alpha
                {
                    out.create(tile_w_nopad * scale, tile_h_nopad * scale, channels);
                    if (scale == 4)
                    {
                        for (int q = 0; q < 3; q++)
                        {
                            float* outptr = out.channel(q);

                            for (int i = 0; i < out.h; i++)
                            {
                                const float* inptr = in_tile.channel(q).row(prepadding + i / 4) + prepadding;
                                const float* ptr = out_tile.channel(q).row(i);

                                for (int j = 0; j < out.w; j++)
                                {
                                    float v = *ptr++ + inptr[j / 4];
                                    if (is_pro_model)
                                        v = (v - 0.15f) * (255.f / 0.7f);
                                    else
                                        v = v * 255.f;
                                    *outptr++ = std::max(0.f, std::min(255.f, v + 0.5f));
                                }
                            }
                        }
                    }
                    else
                    {
                        for (int q = 0; q < 3; q++)
                        {
                            float* outptr = out.channel(q);

                            for (int i = 0; i < out.h; i++)
                            {
                                const float* ptr = out_tile.channel(q).row(i);

                                for (int j = 0; j < out.w; j++)
                                {
                                    float v = *ptr++;
                                    if (is_pro_model)
                                        v = (v - 0.15f) * (255.f / 0.7f);
                                    else
                                        v = v * 255.f;
                                    *outptr++ = std::max(0.f, std::min(255.f, v + 0.5f));
                                }
                            }
                        }
                    }

                    if (channels == 4)
                    {
                        memcpy(out.channel_range(3, 1), out_alpha_tile, out_alpha_tile.total() * sizeof(float));
                    }
                }
            }

            {
                if (channels == 3)
                {
                    out.to_pixels((unsigned char*)outimage.data + (size_t)yi * scale * TILE_SIZE_Y * w * scale * channels + (size_t)xi * scale * TILE_SIZE_X * channels, bgr_mode ? ncnn::Mat::PIXEL_RGB2BGR : ncnn::Mat::PIXEL_RGB, w * scale * channels);
                }
                if (channels == 4)
                {
                    out.to_pixels((unsigned char*)outimage.data + (size_t)yi * scale * TILE_SIZE_Y * w * scale * channels + (size_t)xi * scale * TILE_SIZE_X * channels, bgr_mode ? ncnn::Mat::PIXEL_RGBA2BGRA : ncnn::Mat::PIXEL_RGBA, w * scale * channels);
                }
            }
        }
    }

    return 0;
}

int RealCUGAN::process_cpu_se_sync_gap(const ncnn::Mat& inimage, const std::vector<std::string>& names, FeatureCache& cache, int current_tilesize) const
{
    const unsigned char* pixeldata = (const unsigned char*)inimage.data;
    const int w = inimage.w;
    const int h = inimage.h;
    const int channels = inimage.elempack;

    const int TILE_SIZE_X = current_tilesize > 0 ? current_tilesize : tilesize;
    const int TILE_SIZE_Y = current_tilesize > 0 ? current_tilesize : tilesize;

    ncnn::Option opt = net.opt;

    // each tile 400x400
    const int xtiles = (w + TILE_SIZE_X - 1) / TILE_SIZE_X;
    const int ytiles = (h + TILE_SIZE_Y - 1) / TILE_SIZE_Y;

    const int tta_count = (tta_level > 1) ? tta_level : 1;
    const int num_ti = (syncgap >= 2) ? 1 : tta_count;

    std::vector< std::vector< std::vector<ncnn::Mat> > > feats(names.size(), std::vector< std::vector<ncnn::Mat> >(num_ti));
    std::vector<float> tile_weights;
    tile_weights.reserve(ytiles * xtiles);
    float total_weight = 0.f;

    for (int yi = 0; yi < ytiles; yi++)
    {
        const int tile_h_nopad = std::min((yi + 1) * TILE_SIZE_Y, h) - yi * TILE_SIZE_Y;
        for (int xi = 0; xi < xtiles; xi++)
        {
            const int tile_w_nopad = std::min((xi + 1) * TILE_SIZE_X, w) - xi * TILE_SIZE_X;
            const float area = (float)(tile_w_nopad * tile_h_nopad);
            tile_weights.push_back(area);
            total_weight += area;

            for (size_t i = 0; i < names.size(); i++)
            {
                for (int ti = 0; ti < num_ti; ti++)
                {
                    ncnn::Mat feat;
                    cache.load(yi, xi, ti, names[i], feat);

                    feats[i][ti].push_back(feat);
                }
            }
        }
    }

    const int tiles = ytiles * xtiles;

    // global average
    std::vector< std::vector<ncnn::Mat> > avgfeats(names.size(), std::vector<ncnn::Mat>(num_ti));
    for (size_t i = 0; i < names.size(); i++)
    {
        for (int ti = 0; ti < num_ti; ti++)
        {
            // handle feats[i][ti] vector
            {
                ncnn::Mat avgfeat;
                avgfeat.create_like(feats[i][ti][0]);
                avgfeat.fill(0.f);

                int len = avgfeat.total();

                for (int j = 0; j < tiles; j++)
                {
                    const ncnn::Mat f = feats[i][ti][j];
                    const float w_j = tile_weights[j];

                    for (int k = 0; k < len; k++)
                    {
                        avgfeat[k] += f[k] * w_j;
                    }
                }

                if (total_weight > 0.f)
                {
                    for (int k = 0; k < len; k++)
                    {
                        avgfeat[k] /= total_weight;
                    }
                }
                else if (tiles > 0)
                {
                    for (int k = 0; k < len; k++)
                    {
                        avgfeat[k] /= tiles;
                    }
                }

                avgfeats[i][ti] = avgfeat;
            }
        }
    }

    for (int yi = 0; yi < ytiles; yi++)
    {
        for (int xi = 0; xi < xtiles; xi++)
        {
            for (size_t i = 0; i < names.size(); i++)
            {
                for (int ti = 0; ti < tta_count; ti++)
                {
                    int src_ti = (syncgap >= 2) ? 0 : ti;
                    cache.save(yi, xi, ti, names[i], avgfeats[i][src_ti]);
                }
            }
        }
    }

    return 0;
}

int RealCUGAN::process_cpu_se_very_rough_stage0(const ncnn::Mat& inimage, const std::vector<std::string>& names, const std::vector<std::string>& outnames, FeatureCache& cache, int current_tilesize) const
{
    const unsigned char* pixeldata = (const unsigned char*)inimage.data;
    const int w = inimage.w;
    const int h = inimage.h;
    const int channels = inimage.elempack;

    const int TILE_SIZE_X = current_tilesize > 0 ? current_tilesize : tilesize;
    const int TILE_SIZE_Y = current_tilesize > 0 ? current_tilesize : tilesize;

    ncnn::Option opt = net.opt;

    // each tile 400x400
    const int xtiles = (w + TILE_SIZE_X - 1) / TILE_SIZE_X;
    const int ytiles = (h + TILE_SIZE_Y - 1) / TILE_SIZE_Y;

    int yi_start = (ytiles >= 3) ? 1 : 0;
    int xi_start = (xtiles >= 3) ? 1 : 0;

    for (int yi = yi_start; yi < ytiles; yi += 2)
    {
        const int tile_h_nopad = std::min((yi + 1) * TILE_SIZE_Y, h) - yi * TILE_SIZE_Y;

        int prepadding_bottom = prepadding;
        if (scale == 1 || scale == 3)
        {
            prepadding_bottom += (tile_h_nopad + 3) / 4 * 4 - tile_h_nopad;
        }
        if (scale == 2 || scale == 4)
        {
            prepadding_bottom += (tile_h_nopad + 1) / 2 * 2 - tile_h_nopad;
        }

        int in_tile_y0 = std::max(yi * TILE_SIZE_Y - prepadding, 0);
        int in_tile_y1 = std::min((yi + 1) * TILE_SIZE_Y + prepadding_bottom, h);

        for (int xi = xi_start; xi < xtiles; xi += 2)
        {
            const int tile_w_nopad = std::min((xi + 1) * TILE_SIZE_X, w) - xi * TILE_SIZE_X;

            int prepadding_right = prepadding;
            if (scale == 1 || scale == 3)
            {
                prepadding_right += (tile_w_nopad + 3) / 4 * 4 - tile_w_nopad;
            }
            if (scale == 2 || scale == 4)
            {
                prepadding_right += (tile_w_nopad + 1) / 2 * 2 - tile_w_nopad;
            }

            int in_tile_x0 = std::max(xi * TILE_SIZE_X - prepadding, 0);
            int in_tile_x1 = std::min((xi + 1) * TILE_SIZE_X + prepadding_right, w);

            // crop tile
            ncnn::Mat in, in_nopad;
            {
                if (channels == 3)
                {
                    in = ncnn::Mat::from_pixels_roi(pixeldata, bgr_mode ? ncnn::Mat::PIXEL_BGR2RGB : ncnn::Mat::PIXEL_RGB, w, h, in_tile_x0, in_tile_y0, in_tile_x1 - in_tile_x0, in_tile_y1 - in_tile_y0);
                }
                if (channels == 4)
                {
                    in_nopad = ncnn::Mat::from_pixels_roi(pixeldata, bgr_mode ? ncnn::Mat::PIXEL_BGRA2RGBA : ncnn::Mat::PIXEL_RGBA, w, h, xi * TILE_SIZE_X, yi * TILE_SIZE_Y, tile_w_nopad, tile_h_nopad);
                    in = ncnn::Mat::from_pixels_roi(pixeldata, bgr_mode ? ncnn::Mat::PIXEL_BGRA2RGBA : ncnn::Mat::PIXEL_RGBA, w, h, in_tile_x0, in_tile_y0, in_tile_x1 - in_tile_x0, in_tile_y1 - in_tile_y0);
                }
            }

            ncnn::Mat out;

            if (tta_level > 1)
            {
                // split alpha and preproc
                ncnn::Mat in_tile[8];
                ncnn::Mat in_alpha_tile;
                {
                    in_tile[0].create(in.w, in.h, 3);
                    for (int q = 0; q < 3; q++)
                    {
                        const float* ptr = in.channel(q);
                        float* outptr0 = in_tile[0].channel(q);

                        const float norm_scale = is_pro_model ? (0.7f / 255.f) : (1.f / 255.f);
                        const float norm_bias = is_pro_model ? 0.15f : 0.0f;

                        for (int i = 0; i < in.h; i++)
                        {
                            for (int j = 0; j < in.w; j++)
                            {
                                *outptr0++ = *ptr++ * norm_scale + norm_bias;
                            }
                        }
                    }

                    if (channels == 4)
                    {
                        in_alpha_tile = in_nopad.channel_range(3, 1).clone();
                    }
                }

                // border padding
                {
                    int pad_top = std::max(prepadding - yi * TILE_SIZE_Y, 0);
                    int pad_bottom = std::max(std::min((yi + 1) * TILE_SIZE_Y + prepadding_bottom - h, prepadding_bottom), 0);
                    int pad_left = std::max(prepadding - xi * TILE_SIZE_X, 0);
                    int pad_right = std::max(std::min((xi + 1) * TILE_SIZE_X + prepadding_right - w, prepadding_right), 0);

                    ncnn::Mat in_tile_padded;
                    ncnn::copy_make_border(in_tile[0], in_tile_padded, pad_top, pad_bottom, pad_left, pad_right, 2, 0.f, net.opt);
                    in_tile[0] = in_tile_padded;
                }

                // create symmetries up to tta_level
                in_tile[1].create(in_tile[0].w, in_tile[0].h, 3);
                if (tta_level >= 4)
                {
                    in_tile[2].create(in_tile[0].w, in_tile[0].h, 3);
                    in_tile[3].create(in_tile[0].w, in_tile[0].h, 3);
                }
                if (tta_level >= 8)
                {
                    in_tile[4].create(in_tile[0].h, in_tile[0].w, 3);
                    in_tile[5].create(in_tile[0].h, in_tile[0].w, 3);
                    in_tile[6].create(in_tile[0].h, in_tile[0].w, 3);
                    in_tile[7].create(in_tile[0].h, in_tile[0].w, 3);
                }

                for (int q = 0; q < 3; q++)
                {
                    const ncnn::Mat in_tile_0 = in_tile[0].channel(q);
                    ncnn::Mat in_tile_1 = in_tile[1].channel(q);
                    ncnn::Mat in_tile_2 = (tta_level >= 4) ? in_tile[2].channel(q) : ncnn::Mat();
                    ncnn::Mat in_tile_3 = (tta_level >= 4) ? in_tile[3].channel(q) : ncnn::Mat();
                    ncnn::Mat in_tile_4 = (tta_level >= 8) ? in_tile[4].channel(q) : ncnn::Mat();
                    ncnn::Mat in_tile_5 = (tta_level >= 8) ? in_tile[5].channel(q) : ncnn::Mat();
                    ncnn::Mat in_tile_6 = (tta_level >= 8) ? in_tile[6].channel(q) : ncnn::Mat();
                    ncnn::Mat in_tile_7 = (tta_level >= 8) ? in_tile[7].channel(q) : ncnn::Mat();

                    for (int i = 0; i < in_tile[0].h; i++)
                    {
                        const float* outptr0 = in_tile_0.row(i);
                        float* outptr1 = in_tile_1.row(i) + in_tile[0].w - 1;
                        float* outptr2 = (tta_level >= 4) ? (in_tile_2.row(in_tile[0].h - 1 - i) + in_tile[0].w - 1) : nullptr;
                        float* outptr3 = (tta_level >= 4) ? in_tile_3.row(in_tile[0].h - 1 - i) : nullptr;

                        for (int j = 0; j < in_tile[0].w; j++)
                        {
                            float v = *outptr0++;

                            *outptr1-- = v;
                            if (tta_level >= 4)
                            {
                                *outptr2-- = v;
                                *outptr3++ = v;
                            }
                            if (tta_level >= 8)
                            {
                                float* outptr4 = in_tile_4.row(j) + i;
                                float* outptr5 = in_tile_5.row(in_tile[0].w - 1 - j) + i;
                                float* outptr6 = in_tile_6.row(j) + in_tile[0].h - 1 - i;
                                float* outptr7 = in_tile_7.row(in_tile[0].w - 1 - j) + in_tile[0].h - 1 - i;
                                *outptr4 = v;
                                *outptr5 = v;
                                *outptr6 = v;
                                *outptr7 = v;
                            }
                        }
                    }
                }

                // realcugan
                int tta_count = (syncgap >= 2) ? 1 : tta_level;
                for (int ti = 0; ti < tta_count; ti++)
                {
                    ncnn::Extractor ex = net.create_extractor();

                    ex.input("in0", in_tile[ti]);

                    for (size_t i = 0; i < names.size(); i++)
                    {
                        ncnn::Mat feat;
                        cache.load(yi, xi, ti, names[i], feat);

                        ex.input(names[i].c_str(), feat);
                    }

                    for (size_t i = 0; i < outnames.size(); i++)
                    {
                        ncnn::Mat feat;
                        int ret = ex.extract(outnames[i].c_str(), feat);
                        if (ret != 0) return ret;

                        cache.save(yi, xi, ti, outnames[i], feat);
                    }
                }
            }
            else
            {
                // split alpha and preproc
                ncnn::Mat in_tile;
                ncnn::Mat in_alpha_tile;
                {
                    in_tile.create(in.w, in.h, 3);
                    for (int q = 0; q < 3; q++)
                    {
                        const float* ptr = in.channel(q);
                        float* outptr = in_tile.channel(q);

                        const float norm_scale = is_pro_model ? (0.7f / 255.f) : (1.f / 255.f);
                        const float norm_bias = is_pro_model ? 0.15f : 0.0f;

                        for (int i = 0; i < in.w * in.h; i++)
                        {
                            *outptr++ = *ptr++ * norm_scale + norm_bias;
                        }
                    }

                    if (channels == 4)
                    {
                        in_alpha_tile = in_nopad.channel_range(3, 1).clone();
                    }
                }

                // border padding
                {
                    int pad_top = std::max(prepadding - yi * TILE_SIZE_Y, 0);
                    int pad_bottom = std::max(std::min((yi + 1) * TILE_SIZE_Y + prepadding_bottom - h, prepadding_bottom), 0);
                    int pad_left = std::max(prepadding - xi * TILE_SIZE_X, 0);
                    int pad_right = std::max(std::min((xi + 1) * TILE_SIZE_X + prepadding_right - w, prepadding_right), 0);

                    ncnn::Mat in_tile_padded;
                    ncnn::copy_make_border(in_tile, in_tile_padded, pad_top, pad_bottom, pad_left, pad_right, 2, 0.f, net.opt);
                    in_tile = in_tile_padded;
                }

                {
                    ncnn::Extractor ex = net.create_extractor();

                    ex.input("in0", in_tile);

                    for (size_t i = 0; i < names.size(); i++)
                    {
                        ncnn::Mat feat;
                        cache.load(yi, xi, 0, names[i], feat);

                        ex.input(names[i].c_str(), feat);
                    }

                    for (size_t i = 0; i < outnames.size(); i++)
                    {
                        ncnn::Mat feat;
                        int ret = ex.extract(outnames[i].c_str(), feat);
                        if (ret != 0) return ret;

                        cache.save(yi, xi, 0, outnames[i], feat);
                    }
                }
            }
        }
    }

    return 0;
}

int RealCUGAN::process_cpu_se_very_rough_sync_gap(const ncnn::Mat& inimage, const std::vector<std::string>& names, FeatureCache& cache, int current_tilesize) const
{
    const unsigned char* pixeldata = (const unsigned char*)inimage.data;
    const int w = inimage.w;
    const int h = inimage.h;
    const int channels = inimage.elempack;

    const int TILE_SIZE_X = current_tilesize > 0 ? current_tilesize : tilesize;
    const int TILE_SIZE_Y = current_tilesize > 0 ? current_tilesize : tilesize;

    ncnn::Option opt = net.opt;

    // each tile 400x400
    const int xtiles = (w + TILE_SIZE_X - 1) / TILE_SIZE_X;
    const int ytiles = (h + TILE_SIZE_Y - 1) / TILE_SIZE_Y;

    std::vector< std::vector<ncnn::Mat> > feats(names.size());
    std::vector<float> tile_weights;
    float total_weight = 0.f;

    int yi_start = (ytiles >= 3) ? 1 : 0;
    int xi_start = (xtiles >= 3) ? 1 : 0;

    for (int yi = yi_start; yi < ytiles; yi += 2)
    {
        const int tile_h_nopad = std::min((yi + 1) * TILE_SIZE_Y, h) - yi * TILE_SIZE_Y;
        for (int xi = xi_start; xi < xtiles; xi += 2)
        {
            const int tile_w_nopad = std::min((xi + 1) * TILE_SIZE_X, w) - xi * TILE_SIZE_X;
            const float area = (float)(tile_w_nopad * tile_h_nopad);
            tile_weights.push_back(area);
            total_weight += area;

            for (size_t i = 0; i < names.size(); i++)
            {
                ncnn::Mat feat;
                cache.load(yi, xi, 0, names[i], feat);

                feats[i].push_back(feat);
            }
        }
    }

    const int tiles = (int)tile_weights.size();

    // global average
    std::vector<ncnn::Mat> avgfeats(names.size());
    for (size_t i = 0; i < names.size(); i++)
    {
        // handle feats[i] vector
        {
            ncnn::Mat avgfeat;
            avgfeat.create_like(feats[i][0]);
            avgfeat.fill(0.f);

            int len = avgfeat.total();

            for (int j = 0; j < tiles; j++)
            {
                const ncnn::Mat f = feats[i][j];
                const float w_j = tile_weights[j];

                for (int k = 0; k < len; k++)
                {
                    avgfeat[k] += f[k] * w_j;
                }
            }

            if (total_weight > 0.f)
            {
                for (int k = 0; k < len; k++)
                {
                    avgfeat[k] /= total_weight;
                }
            }
            else if (tiles > 0)
            {
                for (int k = 0; k < len; k++)
                {
                    avgfeat[k] /= tiles;
                }
            }

            avgfeats[i] = avgfeat;
        }
    }

    const int tta_count = (tta_level > 1) ? tta_level : 1;
    for (int yi = 0; yi < ytiles; yi++)
    {
        for (int xi = 0; xi < xtiles; xi++)
        {
            for (size_t i = 0; i < names.size(); i++)
            {
                for (int ti = 0; ti < tta_count; ti++)
                {
                    cache.save(yi, xi, ti, names[i], avgfeats[i]);
                }
            }
        }
    }

    return 0;
}
