

#
input_file = "test.txt"
output_file = "Amazon_sport.test.inter"
input_file_test = "valid.txt"
output_file_test = "Amazon_sport.valid.inter"

def process(input_file,output_file):
    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        
        outfile.write("user_id:token\titem_id:token\trating:float\n")

        
        for line in infile:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue  

            user_id = parts[0]  
            positive_item_id = parts[1]  
            outfile.write(f"{user_id}\t{positive_item_id}\t1.0\n")
process(input_file,output_file)
process(input_file_test,output_file_test)



#
#
def process_d1_train(input_file, output_file, d1_user_num):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        outfile.write("user_id:token\titem_id:token\trating:float\n")
        next(infile)
        for line in infile:
            user_id, item_id, rating = line.strip().split('\t')
           
            item_id = str(int(item_id) + d1_user_num)
            outfile.write(f"{user_id}\t{item_id}\t{rating}\n")



def process_d2_train(input_file, output_file, d1_user_num, d1_item_num, d2_user_num, shared_num):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        outfile.write("user_id:token\titem_id:token\trating:float\n")
        next(infile)
        for line in infile:
            user_id, item_id, rating = line.strip().split('\t')
            user_id = int(user_id)
            item_id = int(item_id)

            if user_id >= shared_num:
                user_id += d1_user_num + d1_item_num - shared_num
            item_id += d1_user_num + d1_item_num + d2_user_num - shared_num

            outfile.write(f"{user_id}\t{item_id}\t{rating}\n")



def process_d1_test(input_file, output_file, d1_user_num):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            line = line.strip()
            if not line:  
                continue
            data = line.split('\t')
            user_id = data[0]
            
            item_ids = [str(int(item_id) + d1_user_num) for item_id in data[1:]]

            outfile.write(f"{user_id}\t" + "\t".join(item_ids) + "\n")


def process_d2_test(input_file, output_file, d1_user_num, d1_item_num, d2_user_num, shared_num):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            line = line.strip()
            if not line:  
                continue
            data = line.split('\t')
            user_id = int(data[0])
            item_ids = [int(item_id) for item_id in data[1:]]


            if user_id >= shared_num:
                user_id += d1_user_num + d1_item_num - shared_num
            item_ids = [str(item_id + d1_user_num + d1_item_num + d2_user_num - shared_num) for item_id in item_ids]

            
            outfile.write(f"{user_id}\t" + "\t".join(item_ids) + "\n")

def process_user(input_file, output_file, d1_user_num, d1_item_num, shared_num):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            data = line.strip().split(' ')
            user_id = int(data[0])
            user_src = int(data[1])
            #rating = data[2]
            #user_id, user_src, rating = line.strip().split(' ')
            if int(user_id) >= shared_num:
                user_id += d1_user_num + d1_item_num - shared_num

            outfile.write(f"{user_id}\t{user_src}\t{1}\n")

def src_process_user(input_file, output_file, d1_user_num, d1_item_num, shared_num):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            data = line.strip().split(' ')
            user_id = int(data[1])
            user_src = int(data[0])
            rating = data[2]
            #user_id, user_src, rating = line.strip().split(' ')
            if int(user_id) >= shared_num:
                user_id += d1_user_num + d1_item_num - shared_num

            outfile.write(f"{user_src}\t{user_id}\t{rating}\n")

input_d1_train = "train.txt"
input_d2_train = "train.txt"
input_d1_test_neg = "test.txt"
input_d2_test_neg = "test.txt"
input_d1_valid_neg ="valid.txt"
input_d2_valid_neg = "valid.txt"
input_d1_test = "Amazon_cloth.test.inter"
input_d2_test = "Amazon_sport.test.inter"
input_d1_valid = "Amazon_cloth.valid.inter"
input_d2_valid = "Amazon_sport.valid.inter"


output_d1_train = "Amazon_cloth.train.inter"
output_d2_train = 'Amazon_sport.train.inter"
output_d1_test_neg = "test.txt"
output_d2_test_neg = "test.txt"
output_d1_valid_neg = "valid.txt"
output_d2_valid_neg = "valid.txt"
output_d1_test = "Amazon_cloth.test.inter"
output_d2_test = "Amazon_sport.test.inter"
output_d1_valid ="Amazon_cloth.valid.inter"
output_d2_valid = "Amazon_sport.valid.inter"


#office_arts
# d1_user_num = 15490
# d2_user_num = 9404
# d1_item_num = 6774
# d2_item_num = 5910
# shared_num = 1243
#movie-music
# d1_user_num = 38064
# d2_user_num = 15404
# d1_item_num = 17125
# d2_item_num = 11525
# shared_num = 1926
#elec_cell
# d1_user_num = 115575
# d2_user_num = 44983
# d1_item_num = 40334
# d2_item_num = 19355
# shared_num = 11617
# cloth_sport
# d1_user_num = 143465
# d2_user_num = 62719
# d1_item_num = 56780
# d2_item_num = 29865
# shared_num = 9419
# toy_video
# d1_user_num = 34559
# d2_user_num = 17460
# d1_item_num = 19003
# d2_item_num = 7323
# shared_num = 1064
# movie_book douban
# d1_user_num = 2642
# d2_user_num = 1294
# d1_item_num = 20560
# d2_item_num = 7089
# shared_num = 1280
#movie-movielens
# d1_user_num = 38064
# d2_user_num = 6035
# d1_item_num = 17125
# d2_item_num = 3416
# shared_num = 0
# new
#elec-cell
d1_user_num = 32109
d2_user_num = 10242
d1_item_num = 12868
d2_item_num = 5755
shared_num = 711

process_d1_train(input_d1_train, output_d1_train, d1_user_num)
process_d1_train(input_d1_test, output_d1_test, d1_user_num)
process_d1_train(input_d1_valid, output_d1_valid, d1_user_num)
process_d2_train(input_d2_train, output_d2_train, d1_user_num, d1_item_num, d2_user_num, shared_num)
process_d2_train(input_d2_test, output_d2_test, d1_user_num, d1_item_num, d2_user_num, shared_num)
process_d2_train(input_d2_valid, output_d2_valid, d1_user_num, d1_item_num, d2_user_num, shared_num)
process_d1_test(input_d1_test_neg, output_d1_test_neg, d1_user_num)
process_d2_test(input_d2_test_neg, output_d2_test_neg, d1_user_num, d1_item_num, d2_user_num, shared_num)
process_d1_test(input_d1_valid_neg, output_d1_valid_neg, d1_user_num)
process_d2_test(input_d2_valid_neg, output_d2_valid_neg, d1_user_num, d1_item_num, d2_user_num, shared_num)
