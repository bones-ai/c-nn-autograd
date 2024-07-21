#include<stdio.h>
#include <time.h>
#include <stdlib.h>
#include "nn.h"

void train()
{
    size_t num_iterations = 500;
    double step_size = 0.0001;
    double loss;
    size_t arch[] = {3, 4, 4, 1};

    NN *nn = nn_create(arch, 4);
    for (int n = 0; n <= num_iterations; n ++)
    {
        Value *inputs[][3] = {
            { value_create(2.0), value_create(3.0), value_create(-1.0) },
            { value_create(3.0), value_create(-1.0), value_create(0.5) },
            { value_create(0.5), value_create(1.0), value_create(1.0) },
            { value_create(1.0), value_create(1.0), value_create(-1.0) },
        };
        Value *outputs[][1] = {
            { value_create(1.0) },
            { value_create(-1.0) },
            { value_create(-1.0) },
            { value_create(1.0) },
        };

        size_t nn_inputs_size = sizeof(inputs[0]) / sizeof(inputs[0][0]);
        size_t num_inputs = sizeof(inputs) / sizeof(inputs[0]);
        size_t nn_output_size = sizeof(outputs[0]) / sizeof(outputs[0][0]);
        size_t num_layers = sizeof(arch) / sizeof(size_t);

        // Forward pass + loss calc
        Value* total_loss = value_create(0.0);
        for (int i = 0; i < num_inputs; i++)
        {
            Value **res = nn_forward(nn, inputs[i]);
            for (int j = 0; j < nn_output_size; j++)
            {
                // loss += (expected - actual) ** 2
                Value* neg_actual = value_operation(res[j], &(Value){-1.0}, MUL);
                Value* diff = value_operation(outputs[i][j], neg_actual, ADD);
                Value* sq = value_operation(diff, diff, MUL);
                total_loss = value_operation(total_loss, sq, ADD);
            }
            free(res);
        }

        // Calc gradients and backprop
        value_init_backprop(total_loss);
        value_init_nudge(total_loss, step_size);
        loss = total_loss->data;
        value_free(total_loss);

        if (n % 10 == 0)
            printf("Step: %d, Loss: %f\n", n, loss);

        // Free inputs and outputs
        for (int i = 0; i < num_inputs; i++)
        {
            for (int j = 0; j < nn_inputs_size; j++)
            {
                value_free(inputs[i][j]);
            }
            value_free(outputs[i][0]);
        }
    }

    nn_free(nn);
}

int main()
{
    // Seed rand
    srand(time(NULL));

    train();
    return 0;
}
