# ttnt
1. Mục tiêu của thuật toán

Tô màu các đỉnh của một đồ thị sao cho:

Hai đỉnh kề nhau không có cùng màu

Sử dụng ít màu nhất có thể (theo cách tham lam, không đảm bảo tối ưu tuyệt đối)

Dữ liệu đầu vào là ma trận kề đọc từ file graph.txt.

2. Đọc dữ liệu đồ thị
def read_adj_matrix(filename):


Đọc ma trận kề từ file

Mỗi dòng trong file tương ứng với một đỉnh

Giá trị:

1 → có cạnh nối

0 → không có cạnh

Ví dụ (6 đỉnh):

0 1 1 0 0 1
1 0 1 1 0 0
...


Kết quả:

G = [
  [0,1,1,0,0,1],
  [1,0,1,1,0,0],
  ...
]

3. Ánh xạ tên đỉnh → chỉ số
node = "ABCDEF"
t_ = {}


Gán tên cho các đỉnh:

A → 0

B → 1

...

Mục đích: dễ truy cập ma trận kề bằng tên đỉnh

t_['A'] = 0
t_['B'] = 1

4. Tính bậc của các đỉnh
degree.append(sum(G[i]))


Bậc của đỉnh = tổng số cạnh nối với đỉnh đó

Trong ma trận kề:

Bậc đỉnh i = tổng dòng i

Ví dụ:

degree = [3, 2, 4, 1, 2, 3]

5. Khởi tạo tập màu cho mỗi đỉnh
colorDict[node[i]] = ["Blue", "Red", "Yellow", "Green"]


Ban đầu, mỗi đỉnh đều có thể dùng tất cả các màu

colorDict lưu các màu chưa bị cấm của từng đỉnh

6. Sắp xếp đỉnh theo bậc giảm dần
Ý tưởng

Đỉnh có bậc lớn tô trước → giảm xung đột về sau

Thực hiện bằng Selection Sort

sortedNode = ['C', 'A', 'F', 'B', 'E', 'D']


Ví dụ:

Đỉnh	Bậc
C	    4
A	    3
F	    3
B	    2
E	    2
D	    1
7. Thuật toán tô màu (Greedy Coloring)
Vòng lặp chính
for n in sortedNode:

Bước 1: Gán màu đầu tiên có thể
theSolution[n] = setTheColor[0]


Chọn màu đầu tiên trong danh sách màu hợp lệ

Đây là chiến lược tham lam

Bước 2: Loại màu này khỏi các đỉnh kề
if adjacentNode[j] == 1:
    colorDict[node[j]].remove(setTheColor[0])


Nếu đỉnh n kề với đỉnh j

Màu vừa dùng không được phép dùng cho đỉnh kề

→ loại màu đó khỏi danh sách màu của đỉnh kề

Điều này đảm bảo:

Hai đỉnh kề nhau không bao giờ có cùng màu

8. In và ghi kết quả
print("Đỉnh A = Blue")


Và ghi ra file:

Đỉnh A = Blue
Đỉnh B = Red
...

9. Bản chất thuật toán
Thuộc tính	Mô tả
Loại thuật toán	Tham lam (Greedy)
Dựa trên	Welsh–Powell (bậc giảm dần)
Đảm bảo đúng	Có
Đảm bảo tối ưu số màu	Không
Độ phức tạp	O(n²)


