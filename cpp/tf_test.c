#include <stdlib.h>

#include "tf_data.h"

int main() {
  printf("Tensorflow C API version %s\n", TF_Version());

  const char* modelName = "test_model";
  const char* inputs[] = {"serving_default_MyInput1", "serving_default_MyInput2"};

  printf("Loading model from '%s'...\n", modelName);

  Model* model = loadModel(modelName, inputs, 2, 0);

  float data1[] = {1, 2, 3, 4};
  int64_t dims1[] = {2, 2};
  float data2[] = {10, 10};
  int64_t dims2[] = {2, 1};

  NumericInputs* input1 = createNumericInputs(data1, 2, dims1);
  NumericInputs* input2 = createNumericInputs(data2, 2, dims2);
  NumericInputs* allInputs[] = {input1, input2};

  Outputs* output = runModel(model, allInputs, NULL);

  printOutput(output);
  deleteOutputs(output);
  deleteModel(model);

  return 0;
}
