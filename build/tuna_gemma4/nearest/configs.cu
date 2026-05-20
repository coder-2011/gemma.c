#define TRANSFORM_N true
#define TRANSFORM_T false
#define LAYOUT_C true
#define LAYOUT_F false
#define QUANTIZED true
#define NOT_QUANTIZED false

typedef bool MATRIX_QUANTIZE_STATUS;
typedef bool MATRIX_TRANSFORM;

struct KernelConfig
{
    const int tileM;
    const int tileN;
    const int tileK;
    const int patchM;
    const int patchN;
    const int k;
    const int m;
    const int splitK;
    const int warpCountM;
    const int warpCountN;
    const int mmaSizeM;
    const int mmaSizeN;
    const int mmaSizeK;
    const int warpMmaCountM;
    const int warpMmaCountN;
    const int warpMmaCountK;
    const int contiguousBytesA;
    const int contiguousBytesB;
    const int deqBlockSize;
    const int stages;
    const int absMaxPerBlock;
    const int threadsPerBlock;
    const int pipelineStrat;
    const int paddingC;

    static constexpr int codeSize = 16;
    static constexpr bool isSafe = false;
    static constexpr int alignSizeBytesA = 16;
    static constexpr int alignSizeBytesB = 16;
    static constexpr MATRIX_TRANSFORM transformA = TRANSFORM_N;
    static constexpr MATRIX_TRANSFORM transformB = TRANSFORM_T;
    static constexpr MATRIX_TRANSFORM transformC = TRANSFORM_T;
    static constexpr MATRIX_QUANTIZE_STATUS quantStatA = QUANTIZED;
    static constexpr MATRIX_QUANTIZE_STATUS quantStatB = NOT_QUANTIZED;
    static constexpr int deqBlockCount = 1;
};

constexpr KernelConfig gemma4_ffn_gate_up_128 = {
    /* tileM */ 128,
    /* tileN */ 64,
    /* tileK */ 32,
    /* patchM */ 2,
    /* patchN */ 1,
    /* k */ 5376,
    /* m */ 43008,
    /* splitK */ 1,
    /* warpCountM */ 4,
    /* warpCountN */ 1,
    /* mmaSizeM */ 16,
    /* mmaSizeN */ 8,
    /* mmaSizeK */ 16,
    /* warpMmaCountM */ 2,
    /* warpMmaCountN */ 8,
    /* warpMmaCountK */ 2,
    /* contiguousBytesA */ -1,
    /* contiguousBytesB */ 64,
    /* deqBlockSize */ 5376,
    /* stages */ 4,
    /* absMaxPerBlock */ 128,
    /* threadsPerBlock */ 128,
    /* pipelineStrat */ 1,
    /* paddingC */ 2};

