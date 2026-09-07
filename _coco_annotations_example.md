Dưới đây là toàn bộ nội dung kèm **comment trực tiếp từng dòng** để hiểu rõ tác dụng của từng trường (có thể nạp trực tiếp bằng `json.load()` trong Python không lỗi):

```jsonc
{
  // =========================================================================
  // 1. KHỐI THÔNG TIN CHUNG (Optional - Có hay không cũng không ảnh hưởng train)
  // =========================================================================
  "info": {
    "description": "Dataset mau danh cho MedSAM3 LoRA Finetune", // Mô tả ngắn gọn về tập dữ liệu
    "version": "1.0",                                          // Phiên bản dữ liệu
    "year": 2026,                                              // Năm tạo dữ liệu
    "contributor": "User"                                      // Người tạo/gắn nhãn
  },

  // =========================================================================
  // 2. KHỐI DANH MỤC NHÃN (BẮT BUỘC)
  // =========================================================================
  "categories": [
    {
      "id": 1,                      // ID số của nhãn (bắt đầu từ 1)
      "name": "defect",             // TÊN NHÃN: ĐÂY LÀ TEXT PROMPT ĐƯA VÀO SAM3 TEXT ENCODER
      "supercategory": "defect"     // Nhóm cha (tùy chọn, mô hình không bắt buộc đọc)
    },
    {
      "id": 2,                      // ID số của nhãn thứ 2
      "name": "crack",              // Text prompt cho loại lỗi thứ 2: "crack"
      "supercategory": "defect"     // Cùng thuộc nhóm lỗi
    }
  ],

  // =========================================================================
  // 3. KHỐI DANH SÁCH ẢNH (BẮT BUỘC)
  // =========================================================================
  "images": [
    {
      "id": 0,                      // ID số duy nhất của bức ảnh đầu tiên (thường từ 0)
      "file_name": "img001.jpg",    // Tên file ảnh thực tế nằm trong folder train/ hoặc valid/
      "width": 640,                 // Chiều rộng pixel ảnh gốc (dùng để scale box/mask)
      "height": 480                 // Chiều cao pixel ảnh gốc
    },
    {
      "id": 1,                      // ID của bức ảnh thứ hai
      "file_name": "img002.jpg",    // Tên file ảnh thứ hai
      "width": 1008,                // Chiều rộng ảnh thứ hai
      "height": 1008                // Chiều cao ảnh thứ hai
    }
  ],

  // =========================================================================
  // 4. KHỐI ANNOTATIONS - CHI TIẾT TỪNG VẬT THỂ / VÙNG PHÂN ĐOẠN (BẮT BUỘC)
  // =========================================================================
  "annotations": [
    // --- VẬT THỂ THỨ 1 (Nằm trên ảnh id: 0, thuộc nhãn category_id: 1) ---
    {
      "id": 1,                      // ID định danh duy nhất của vùng đánh nhãn này
      "image_id": 0,                // Nằm trên ảnh id: 0 ("img001.jpg")
      "category_id": 1,             // Thuộc nhãn id: 1 ("defect")
      "bbox": [                     // Bounding box bao quanh vật thể dạng [x, y, w, h]
        100.0,                      // x: Tọa độ pixel góc trên bên trái
        150.0,                      // y: Tọa độ pixel góc trên bên trái
        80.0,                       // width: Chiều rộng của hộp bao
        60.0                        // height: Chiều cao của hộp bao
      ],
      "area": 4800.0,               // Diện tích vùng mask (80 x 60 = 4800 pixel)
      "segmentation": [             // MẶT NẠ PHÂN ĐOẠN (DẠNG ĐA GIÁC POLYGON)
        [
          100.0, 150.0,             // Điểm 1: góc trên-trái (x1, y1)
          180.0, 150.0,             // Điểm 2: góc trên-phải (x2, y2)
          180.0, 210.0,             // Điểm 3: góc dưới-phải (x3, y3)
          100.0, 210.0              // Điểm 4: góc dưới-trái (x4, y4) khép kín viền
        ]
      ],
      "iscrowd": 0                  // 0 = vật thể đơn lẻ (mặc định), 1 = đám đông dính chùm
    },

    // --- VẬT THỂ THỨ 2 (Nằm trên ảnh id: 1, thuộc nhãn category_id: 2) ---
    {
      "id": 2,                      // ID định danh của vùng đánh nhãn thứ 2
      "image_id": 1,                // Nằm trên ảnh id: 1 ("img002.jpg")
      "category_id": 2,             // Thuộc nhãn id: 2 ("crack")
      "bbox": [
        200.0,                      // x: 200 pixel
        300.0,                      // y: 300 pixel
        50.0,                       // width: 50 pixel
        120.0                       // height: 120 pixel
      ],
      "area": 6000.0,               // Diện tích mask
      "segmentation": [             // Tọa độ polygon khép kín vết nứt
        [
          200.0, 300.0,
          250.0, 300.0,
          250.0, 420.0,
          200.0, 420.0
        ]
      ],
      "iscrowd": 0                  // Đối tượng riêng lẻ
    }
  ]
}
```

---

### 3 Điểm mấu chốt cần nhớ khi tự tạo hoặc sửa file này:
1. **Liên kết ID:** `annotations[i].image_id` bắt buộc phải trùng với một `images[j].id`, và `annotations[i].category_id` bắt buộc phải trùng với một `categories[k].id`.
2. **Text Prompt:** Khi huấn luyện, code sẽ tự động ghép:
   * Ảnh `img001.jpg` $\rightarrow$ Học nhận diện với text prompt: `"defect"` (từ category 1).
   * Ảnh `img002.jpg` $\rightarrow$ Học nhận diện với text prompt: `"crack"` (từ category 2).
3. **Segmentation Polygon:** Mảng con `[x1, y1, x2, y2, ...]` phải có **số lượng phần tử chẵn** (vì là các cặp tọa độ $(x, y)$ liên tiếp tạo thành đa giác viền quanh đối tượng).