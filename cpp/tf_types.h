#include <stdlib.h>

int dataLength(int64_t* dims, int nDims) {
    if (nDims > 0) {
        int dLen = 1;
        for (int i = 0; i < nDims; i++) dLen *= dims[i];

        return dLen;
    }
    else return 0;
}

typedef struct ModelSummary {
    TF_Session* session;
    TF_Graph* graph;
    TF_Output* feeds;
    TF_Output* fetches;
    int numericInputs;
    int stringInputs;
    int totalInputs;
} Model;

void deleteModel(Model model) {
    TF_Status* status = TF_NewStatus();

    TF_DeleteSession(model.session, status);
    TF_DeleteGraph(model.graph);
    TF_DeleteStatus(status);

    if (model.feeds != NULL) free(model.feeds);
    if (model.fetches != NULL) free(model.fetches);
}

typedef struct ModelNumericOutputs {
    float* data;
    int nDims;
    int64_t* dims;
    char* statusMessage;
    int statusCode;
} Outputs;

void deleteOutputs(Outputs output) {
    if (output.data != NULL) free(output.data);
    if (output.statusMessage != NULL) free(output.statusMessage);
    if (output.dims != NULL) free(output.dims);
}

typedef struct ModelNumericInputs {
    float* data;
    int nDims;
    int64_t* dims;
} NumericInputs;

NumericInputs createNumericInputs(float* data, int nDims, int64_t* dims) {
    NumericInputs input;
    input.data = data;
    input.nDims = nDims;
    input.dims = dims;

    return input;
}

void deleteNumericInputs(NumericInputs input) {
    TF_Status* status = TF_NewStatus();

    if (input.data != NULL) free(input.data);
    if (input.dims != NULL) free(input.dims);
}

typedef struct ModelStringInputs {
    char** data;
    int nDims;
    int64_t* dims;
} StringInputs;

void deleteStringInputs(StringInputs input) {
    TF_Status* status = TF_NewStatus();

    if (input.nDims > 0) {
        int nStrings = dataLength(input.dims, input.nDims);

        for (int i = 0; i < nStrings; i++)
            if (input.data[i] != NULL) free(input.data[i]);

        if (input.data != NULL) free(input.data);
        if (input.dims != NULL) free(input.dims);
    }
}
