#ifndef WEBP_IMAGE_H
#define WEBP_IMAGE_H

// webp image decoder and encoder with libwebp
#include <stdio.h>
#include <stdlib.h>
#include "webp/decode.h"
#include "webp/encode.h"

unsigned char* webp_load(const unsigned char* buffer, int len, int* w, int* h, int* c)
{
    unsigned char* pixeldata = 0;

    WebPDecoderConfig config;
    WebPInitDecoderConfig(&config);

    if (WebPGetFeatures(buffer, len, &config.input) != VP8_STATUS_OK)
        return NULL;

    int width = config.input.width;
    int height = config.input.height;
    int channels = config.input.has_alpha ? 4 : 3;

    pixeldata = (unsigned char*)malloc(width * height * channels);

#if _WIN32
    config.output.colorspace = channels == 4 ? MODE_BGRA : MODE_BGR;
#else
    config.output.colorspace = channels == 4 ? MODE_RGBA : MODE_RGB;
#endif

    config.output.u.RGBA.stride = width * channels;
    config.output.u.RGBA.size = width * height * channels;
    config.output.u.RGBA.rgba = pixeldata;
    config.output.is_external_memory = 1;

    if (WebPDecode(buffer, len, &config) != VP8_STATUS_OK)
    {
        free(pixeldata);
        return NULL;
    }

    *w = width;
    *h = height;
    *c = channels;

    return pixeldata;
}

enum WebPFormat {
    WEBP_FMT_AUTO = 0,
    WEBP_FMT_RGB,
    WEBP_FMT_BGR,
    WEBP_FMT_RGBA,
    WEBP_FMT_BGRA
};

#if _WIN32
int webp_save(const wchar_t* filepath, int w, int h, int c, const unsigned char* pixeldata, WebPFormat format = WEBP_FMT_AUTO)
#else
int webp_save(const char* filepath, int w, int h, int c, const unsigned char* pixeldata, WebPFormat format = WEBP_FMT_AUTO)
#endif
{
    int ret = 0;
    FILE* fp = 0;

    if (c != 3 && c != 4)
        return 0;

    WebPConfig config;
    if (!WebPConfigInit(&config))
        return 0;

    config.lossless = 1;
    config.exact = 1;
    config.quality = 75.0f;
    config.thread_level = 1;

    if (!WebPValidateConfig(&config))
        return 0;

    WebPPicture picture;
    if (!WebPPictureInit(&picture))
        return 0;

    picture.width = w;
    picture.height = h;
    picture.use_argb = 1;

    WebPMemoryWriter writer;
    WebPMemoryWriterInit(&writer);
    picture.writer = WebPMemoryWrite;
    picture.custom_ptr = &writer;

    if (format == WEBP_FMT_AUTO)
    {
#if _WIN32
        format = (c == 4) ? WEBP_FMT_BGRA : WEBP_FMT_BGR;
#else
        format = (c == 4) ? WEBP_FMT_RGBA : WEBP_FMT_RGB;
#endif
    }

    int import_ok = 0;
    switch (format)
    {
    case WEBP_FMT_RGB:
        import_ok = WebPPictureImportRGB(&picture, pixeldata, w * 3);
        break;
    case WEBP_FMT_BGR:
        import_ok = WebPPictureImportBGR(&picture, pixeldata, w * 3);
        break;
    case WEBP_FMT_RGBA:
        import_ok = WebPPictureImportRGBA(&picture, pixeldata, w * 4);
        break;
    case WEBP_FMT_BGRA:
        import_ok = WebPPictureImportBGRA(&picture, pixeldata, w * 4);
        break;
    default:
        break;
    }

    if (!import_ok)
    {
        WebPPictureFree(&picture);
        WebPMemoryWriterClear(&writer);
        return 0;
    }

    if (!WebPEncode(&config, &picture))
    {
        WebPPictureFree(&picture);
        WebPMemoryWriterClear(&writer);
        return 0;
    }

#if _WIN32
    fp = _wfopen(filepath, L"wb");
#else
    fp = fopen(filepath, "wb");
#endif
    if (fp && writer.mem && writer.size > 0)
    {
        if (fwrite(writer.mem, 1, writer.size, fp) == writer.size)
        {
            ret = 1;
        }
    }

    if (fp) fclose(fp);
    WebPPictureFree(&picture);
    WebPMemoryWriterClear(&writer);

    return ret;
}

#endif // WEBP_IMAGE_H
