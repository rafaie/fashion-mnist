

models_count = 2
learning_rates_count = 2
regularizers_count = 2
drop_state_count = 4
batch_sizes_count = 1


if __name__ == "__main__":
    for i in range(batch_sizes_count):
        for j in range(drop_state_count):
            for k in range(regularizers_count):
                for l in range(learning_rates_count):
                    for m in range(models_count):
                        id = str(i) + str(j) + str(k) + str(l) + str(m)

                        # # print(id)
                        # if i == 0 :
                        #     print("model_"+id+',', ' batch_sizes_40')
                        # else:
                        #     print("model_"+id+',', ' batch_sizes_20')

                        if j == 0 :
                           print("model_"+id+', Dropout,', ' Drop_state_none')
                        elif j == 1:
                           print("model_"+id+', Dropout,', ' Drop_state_beginning')
                        elif j == 2:
                           print("model_"+id+', Dropout,', ' Drop_state_middle')
                        elif j == 3:
                           print("model_"+id+', Dropout,', ' Drop_state_end')

                        if k == 0 :
                            print("model_"+id+', Regularizer,', ' Regularizer_None')
                        else:
                            print("model_"+id+',Regularizer,', ' Regularizers_L2')

                        if l == 0 :
                            print("model_"+id+', Learning_rate,', ' Learning_rate_0.0001')
                        else:
                            print("model_"+id+', Learning_rate,', ' Learning_rate_0.00001')

                        if m == 0 :
                            print("model_"+id+', Model,', ' Model_3H_800_400_100')
                        else:
                            print("model_"+id+', Model,', ' Model_4H_800_400_400_100')
                           
                        
                        

                             
