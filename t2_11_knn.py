import tkinter as tk
from tkinter import messagebox, ttk
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def euclidean_distance(row1, row2):
    """Tính khoảng cách Euclidean."""
    p1 = np.array(row1)
    p2 = np.array(row2)
    return np.sqrt(np.sum((p1 - p2)**2))

def get_neighbors_with_distances(train_data, test_point):
    """Tìm tất cả các khoảng cách và sắp xếp."""
    distances = []
    for train_row in train_data:
        train_features = train_row[:-1]
        dist = euclidean_distance(train_features, test_point)
        distances.append((train_row, dist)) 
    distances.sort(key=lambda x: x[1])
    return distances

def predict_classification(neighbors):
    """Thực hiện bỏ phiếu đa số."""
    neighbor_classes = [row[-1] for row, dist in neighbors] 
    most_common = Counter(neighbor_classes).most_common(1)
    return most_common[0][0]


SAMPLE_DATASET = [
    [2.78, 2.55, 'Táo'], [1.46, 2.36, 'Táo'], [3.39, 4.40, 'Táo'],
    [1.38, 1.85, 'Táo'], [3.06, 3.00, 'Táo'], 
    [7.62, 2.75, 'Cam'], [5.33, 2.90, 'Cam'], [3.50, 3.23, 'Cam'], 
    [6.91, 1.15, 'Cam'], [7.83, 2.24, 'Cam']
]
FEATURE_NAMES = ["Chiều rộng", "Chiều dài"]
CLASS_COLORS = {'Táo': 'red', 'Cam': 'blue', 'Khác': 'gray'}


class StepByStepKNNApp:
    def __init__(self, master, dataset, feature_names):
        self.master = master
        master.title("Thuật Toán KNN")
        master.geometry("1000x600") 

        self.dataset = dataset
        self.feature_names = feature_names
        self.current_step = 0
        self.all_distances = None
        self.new_point = None
        
        self.create_widgets()

    def create_widgets(self):
        main_pane = ttk.PanedWindow(self.master, orient=tk.HORIZONTAL)
        main_pane.pack(fill="both", expand=True)

        left_frame = ttk.Frame(main_pane, width=400, padding="10")
        main_pane.add(left_frame, weight=0)

        self.plot_frame = ttk.Frame(main_pane, padding="10")
        main_pane.add(self.plot_frame, weight=1)

        input_frame = ttk.LabelFrame(left_frame, text="1. Thiết lập Đầu vào")
        input_frame.pack(fill="x", pady=5)
        
        ttk.Label(input_frame, text="K (Hàng xóm):").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.k_entry = ttk.Entry(input_frame, width=8)
        self.k_entry.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        self.k_entry.insert(0, "3")

        ttk.Label(input_frame, text=f"{self.feature_names[0]}:").grid(row=1, column=0, padx=5, pady=2, sticky="w")
        self.f1_entry = ttk.Entry(input_frame, width=15)
        self.f1_entry.grid(row=1, column=1, padx=5, pady=2, sticky="w")
        self.f1_entry.insert(0, "5.5") 

        ttk.Label(input_frame, text=f"{self.feature_names[1]}:").grid(row=2, column=0, padx=5, pady=2, sticky="w")
        self.f2_entry = ttk.Entry(input_frame, width=15)
        self.f2_entry.grid(row=2, column=1, padx=5, pady=2, sticky="w")
        self.f2_entry.insert(0, "2.5") 
        
        ttk.Button(input_frame, text="Đặt lại", command=self.reset_process).grid(row=0, column=2, rowspan=3, padx=15, sticky="ns")

        control_frame = ttk.Frame(left_frame)
        control_frame.pack(fill="x", pady=10)
        
        self.step_button = ttk.Button(control_frame, text="▶️ BẮT ĐẦU (Bước 1/3)", command=self.next_step)
        self.step_button.pack(fill="x")

        result_frame = ttk.LabelFrame(left_frame, text="3. Trạng thái & Kết quả Chi tiết")
        result_frame.pack(fill="both", expand=True, pady=5)
        
        self.status_label = ttk.Label(result_frame, text="Sẵn sàng. Nhấn START để bắt đầu!", font=("Helvetica", 10, "bold"))
        self.status_label.pack(padx=10, pady=5, anchor="w")

        self.detail_text = tk.Text(result_frame, height=12, state='disabled', wrap='word', font=("Courier", 9))
        self.detail_text.pack(padx=10, pady=5, fill="both", expand=True)
        
        self.plot_initial_data()

    def update_detail_text(self, content):
        """Hàm trợ giúp để cập nhật Text Widget."""
        self.detail_text.config(state='normal')
        self.detail_text.delete('1.0', tk.END)
        self.detail_text.insert(tk.END, content)
        self.detail_text.config(state='disabled')
        
    def plot_initial_data(self):
        """Vẽ đồ thị dữ liệu huấn luyện thô."""
        fig, ax = plt.subplots(figsize=(6, 5))
        data_array = np.array(self.dataset)
        X = data_array[:, :-1].astype(float)
        y = data_array[:, -1]
        
        classes = np.unique(y)
        for cls in classes:
            class_points = X[y == cls]
            ax.scatter(class_points[:, 0], class_points[:, 1], 
                       c=CLASS_COLORS.get(cls, 'gray'), 
                       label=f'Lớp {cls}', marker='o', alpha=0.7)

        ax.set_title("Dữ liệu Huấn luyện Thô", fontsize=12)
        ax.set_xlabel(self.feature_names[0])
        ax.set_ylabel(self.feature_names[1])
        ax.legend(loc='upper right')
        ax.grid(True, linestyle=':', alpha=0.6)
        
        self.update_plot_canvas(fig)

    def update_plot_canvas(self, fig):
        """Cập nhật canvas vẽ đồ thị."""
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
            
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill=tk.BOTH, expand=True)
        canvas.draw()
        
    def plot_knn_step(self, step_title, k, neighbors_with_dist, highlight_k=False, final_class=None):
        """Vẽ đồ thị minh họa từng bước KNN."""
        fig, ax = plt.subplots(figsize=(6, 5))
        data_array = np.array(self.dataset)
        X = data_array[:, :-1].astype(float)
        y = data_array[:, -1]
        
        classes = np.unique(y)
        for cls in classes:
            class_points = X[y == cls]
            ax.scatter(class_points[:, 0], class_points[:, 1], 
                       c=CLASS_COLORS.get(cls, 'gray'), 
                       label=f'Lớp {cls}', marker='o', alpha=0.5)

        ax.scatter(self.new_point[0], self.new_point[1], 
                   c='black', marker='*', s=250, 
                   label='Điểm mới', zorder=5)

        if highlight_k:
            nearest_neighbors = neighbors_with_dist[:k]
            
            distance_to_kth = nearest_neighbors[-1][1]
            circle = plt.Circle((self.new_point[0], self.new_point[1]), distance_to_kth, 
                                color='orange', fill=False, linestyle='--', alpha=0.8)
            ax.add_artist(circle)
            
            for (neighbor_row, dist) in nearest_neighbors:
                n_features = neighbor_row[:-1]
                ax.plot([self.new_point[0], n_features[0]], [self.new_point[1], n_features[1]], 
                        '--', color=CLASS_COLORS.get(neighbor_row[-1]), linewidth=1)
                ax.scatter(n_features[0], n_features[1], 
                           s=180, facecolors='none', edgecolors='black', 
                           linewidths=2, zorder=4)

        title = f"{step_title} (K={k})"
        if final_class:
             title += f"\nKết luận: {final_class}"
             
        ax.set_title(title, fontsize=12)
        ax.set_xlabel(self.feature_names[0])
        ax.set_ylabel(self.feature_names[1])
        ax.legend(loc='upper right')
        ax.grid(True, linestyle=':', alpha=0.6)
        
        self.update_plot_canvas(fig)

    def reset_process(self):
        """Đặt lại toàn bộ quá trình."""
        self.current_step = 0
        self.all_distances = None
        self.new_point = None
        self.status_label.config(text="Sẵn sàng. Nhấn START để bắt đầu!", foreground="black")
        self.step_button.config(text="▶️ BẮT ĐẦU (Bước 1/3)", state='normal')
        self.update_detail_text("Nhập giá trị K và Features mới, sau đó nhấn START.")
        self.plot_initial_data()
        
    def next_step(self):
        """Điều khiển logic chuyển bước."""
        self.current_step += 1
        
        try:
            k = int(self.k_entry.get())
            f1 = float(self.f1_entry.get())
            f2 = float(self.f2_entry.get())
            self.new_point = [f1, f2]
            
            if k <= 0 or k > len(self.dataset):
                messagebox.showerror("Lỗi K", f"K phải là số nguyên dương và nhỏ hơn {len(self.dataset)}.")
                self.reset_process()
                return

        except ValueError:
            messagebox.showerror("Lỗi Đầu vào", "K và Features phải là số hợp lệ.")
            self.reset_process()
            return

        if self.current_step == 1:
            self.status_label.config(text="1️⃣ BƯỚC 1: Tính Khoảng cách Euclidean đến tất cả điểm.", foreground="blue")
            self.all_distances = get_neighbors_with_distances(self.dataset, self.new_point)
            
            content = f"Điểm mới: {self.new_point}\n\n"
            content += "Khoảng cách đến từng điểm:\n"
            for row, dist in self.all_distances:
                content += f"  - {row[:-1]} -> {row[-1]} (Dist: {dist:.4f})\n"
            
            self.update_detail_text(content)
            self.plot_knn_step("1. Tính Khoảng cách", k, self.all_distances, highlight_k=False)
            self.step_button.config(text="➡️ Bước 2/3: Sắp xếp và Chọn K")
            
        elif self.current_step == 2:
            self.status_label.config(text=f"2️⃣ BƯỚC 2: Sắp xếp và Chọn K={k} Hàng xóm Gần nhất.", foreground="orange")
            self.all_distances.sort(key=lambda x: x[1])  
            nearest_neighbors = self.all_distances[:k]
            
            content = f"K={k} Hàng xóm Gần nhất (Đã sắp xếp):\n"
            for i, (row, dist) in enumerate(nearest_neighbors):
                content += f"  {i+1}. {row[:-1]} -> {row[-1]} (Dist: {dist:.4f})\n"
            
            self.update_detail_text(content)
            self.plot_knn_step("2. Chọn K Hàng xóm", k, self.all_distances, highlight_k=True)
            self.step_button.config(text="➡️ Bước 3/3: Bỏ phiếu và Kết luận")
            
        elif self.current_step == 3:
            self.status_label.config(text="3️⃣ BƯỚC 3: Bỏ phiếu đa số và đưa ra kết quả cuối cùng.", foreground="green")
            
            nearest_neighbors = self.all_distances[:k]
            predicted_class = predict_classification(nearest_neighbors)
            
            neighbor_classes = [row[-1] for row, dist in nearest_neighbors]
            class_counts = Counter(neighbor_classes)
            
            vote_details = ", ".join([f"{cls}: {count} phiếu" for cls, count in class_counts.items()])
            
            content = "--- Tóm tắt Bỏ phiếu Đa số ---\n"
            content += f"Các nhãn của K={k} hàng xóm: {neighbor_classes}\n"
            content += f"Kết quả bỏ phiếu: {vote_details}\n\n"
            content += f"🎉 KẾT LUẬN CUỐI CÙNG: **{predicted_class}**"
            
            self.update_detail_text(content)
            self.plot_knn_step("3. Bỏ phiếu Đa số", k, self.all_distances, highlight_k=True, final_class=predicted_class)

            self.step_button.config(text="✅ HOÀN THÀNH. Nhấn Đặt lại để chạy lại.", state='disabled')

        else:
            self.step_button.config(state='disabled')


if __name__ == "__main__":
    root = tk.Tk()
    app = StepByStepKNNApp(root, SAMPLE_DATASET, FEATURE_NAMES)
    root.mainloop()