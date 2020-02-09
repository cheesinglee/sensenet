#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tensorflow/c/c_api.h>

typedef struct ModelSummary {
    TF_Session* session;
    TF_Graph* graph;
    TF_Output* feeds;
    TF_Output* fetches;
} Model;

void deleteModel(Model model) {
    TF_Status* status = TF_NewStatus();

    TF_DeleteSession(model.session, status);
    TF_DeleteGraph(model.graph);
    TF_DeleteStatus(status);

    free(model.feeds);
    free(model.fetches);
}

typedef struct OutputFloats {
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

TF_Output* makeIO(TF_Graph* graph, const char** names, int nInputs) {
    TF_Output* feeds = malloc(sizeof(TF_Input) * nInputs);
    for (int i = 0; i < nInputs; i++) {
        feeds[i].oper = TF_GraphOperationByName(graph, names[i]);
        feeds[i].index = 0;
    }

    return feeds;
}

Model loadModel(const char* exportDir, const char** inputNames, int nInputs) {
    TF_Graph* graph = TF_NewGraph();
    TF_Status* status = TF_NewStatus();
    TF_SessionOptions* sessionOpts = TF_NewSessionOptions();

    const char* tag = "serve";
    const char* outputNames[] = {"PartitionedCall"};

    TF_Session* session = TF_LoadSessionFromSavedModel(sessionOpts,
                                                       NULL,
                                                       exportDir,
                                                       &tag,
                                                       1,
                                                       graph,
                                                       NULL,
                                                       status);


    Model model;

    model.session = session;
    model.graph = graph;
    model.feeds = makeIO(graph, inputNames, nInputs);
    model.fetches = makeIO(graph, outputNames, 1);

    TF_DeleteStatus(status);
    TF_DeleteSessionOptions(sessionOpts);

    return model;
}

TF_Tensor* floatMatrix(const float values[], const int64_t dims[]) {
    int nvals = dims[0] * dims[1];
    int dataSize = sizeof(float) * nvals;

    TF_Tensor* t = TF_AllocateTensor(TF_FLOAT, dims, 2, dataSize);
    memcpy(TF_TensorData(t), values, dataSize);

    return t;
}

Outputs parseOutput(TF_Tensor* outTensor, TF_Status* status) {
    Outputs output;

    output.statusCode = (int)TF_GetCode(status);
    output.statusMessage = NULL;
    output.nDims = 0;
    output.dims = NULL;
    output.data = NULL;

    if (output.statusCode == 0) {
        output.nDims = TF_NumDims(outTensor);
        output.dims = malloc(sizeof(int64_t) * output.nDims);

        size_t dataSize = sizeof(float);

        for (int i = 0; i < output.nDims; i++) {
            output.dims[i] = TF_Dim(outTensor, i);
            dataSize *= output.dims[i];
        }

        output.data = (float*)(malloc(dataSize));
        memcpy(output.data, TF_TensorData(outTensor), dataSize);

        TF_DeleteTensor(outTensor);
    }
    else {
        const char* tfMessage = TF_Message(status);
        output.statusMessage = malloc(sizeof(char) * strlen(tfMessage));

        strcpy(output.statusMessage, tfMessage);
    }

    return output;
}

Outputs runModel(Model model,
                 const float values1[],
                 const int64_t dims1[],
                 const float values2[],
                 const int64_t dims2[]) {

    int nInputs = 2;
    TF_Status* status = TF_NewStatus();

    TF_Tensor* feedValues[] = {
        floatMatrix(values1, dims1),
        floatMatrix(values2, dims2)
    };

    TF_Tensor** fetchValues = malloc(sizeof(TF_Tensor*));

    TF_SessionRun(model.session,
                  NULL,
                  (TF_Output*)(model.feeds),
                  feedValues,
                  nInputs,
                  model.fetches,
                  fetchValues,
                  1,
                  NULL,
                  0,
                  NULL,
                  status);

    for (int i = 0; i < nInputs; i++) {
        TF_DeleteTensor(feedValues[i]);
    }

    Outputs output = parseOutput(fetchValues[0], status);
    TF_DeleteStatus(status);

    return output;
}

void printArray(float* data, int offset, int len) {
    int end = offset + len;

    printf("[");
    for (int i = offset; i < end - 1; i++) {
        printf("%6.2f ", data[i]);
    }
    printf("%6.2f]", data[end - 1]);
}

void printOutput(Outputs output) {
    printf("\n********** Result of Execution ************\n");

    if (output.statusCode != 0) {
        printf("Status code: %d\n", output.statusCode);
        printf("Message: %s", output.statusMessage);
    }
    else {
        if (output.nDims == 1) {
            printArray(output.data, 0, output.dims[0]);
        }
        else if (output.nDims == 2) {
            int start = 0;

            printf("[\n");
            for (int i = 0; i < output.dims[0]; i++) {
                printf("  ");
                printArray(output.data, start, output.dims[1]);
                printf("\n");
                start += output.dims[1];
            }
            printf("]");
        }
    }

    printf("\n");
}

int main() {
  printf("Tensorflow C API version %s\n", TF_Version());

  const char* modelName = "test_model";
  const char* inputs[] = {"serving_default_MyInput1", "serving_default_MyInput2"};

  printf("Loading model from '%s'...\n", modelName);

  Model model = loadModel(modelName, inputs, 2);

  const float v1[] = {1, 2, 3, 4};
  const float v2[] = {10, 10};
  const int64_t d1[] = {2, 2};
  const int64_t d2[] = {2, 1};

  Outputs output = runModel(model, v1, d1, v2, d2);
  printOutput(output);

  deleteOutputs(output);
  // deleteModel(model);

  return 0;
}
