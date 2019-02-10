

models_count = 2
learning_rates_count = 2
regularizers_count = 2
drop_state_count = 4
batch_sizes_count = 2


if __name__ == "__main__":
    for i in range(batch_sizes_count):
        for j in range(drop_state_count):
            for k in range(regularizers_count):
                for l in range(learning_rates_count):
                    for m in range(models_count):
                        id = str(i) + str(j) + str(k) + str(l) + str(m)
                        print(id)
            
