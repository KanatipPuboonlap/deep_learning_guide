# การเลือก Loss Function ใน Deep Learning

การเลือก **Loss Function** เป็นขั้นตอนสำคัญในกระบวนการสร้างโมเดล Deep Learning เนื่องจาก Loss Function กำหนดวิธีที่โมเดลจะวัดความผิดพลาดระหว่างค่าทำนาย (Predicted Output) และค่าจริง (Ground Truth) โดยเป้าหมายของการฝึกโมเดลคือการทำให้ Loss Function มีค่าน้อยที่สุด การเลือก Loss Function ที่เหมาะสมจะช่วยให้โมเดลสามารถเรียนรู้และทำนายผลได้อย่างมีประสิทธิภาพ

---

## 1. เข้าใจบทบาทของ Loss Function

- **Loss Function** ใช้วัดความแตกต่างระหว่างค่าที่โมเดลทำนายออกมาและค่าจริง ซึ่งจะถูกนำมาใช้ในการปรับพารามิเตอร์ของโมเดลผ่านกระบวนการ Backpropagation
- หากเลือก Loss Function ไม่เหมาะสม อาจทำให้โมเดลเรียนรู้ได้ช้าหรือไม่สามารถแก้ปัญหาได้ตามที่ต้องการ

---

## 2. ประเภทของ Loss Function และการใช้งาน

### a. **Mean Squared Error (MSE)**  
- **สูตร**: $ \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 $
- **ข้อดี**:
  - คำนวณง่ายและเหมาะสำหรับงานที่ต้องการลดความผิดพลาดแบบเชิงเส้น
  - ใช้งานได้ดีเมื่อข้อมูลมีการกระจายตัวปกติ (Normal Distribution)
- **ข้อเสีย**:
  - ไวต่อ Outliers เนื่องจากการยกกำลังสองทำให้ค่าผิดพลาดที่มากขึ้นมีผลกระทบสูง
- **การใช้งาน**:
  - เหมาะสำหรับงาน Regression เช่น การทำนายราคาบ้าน, การทำนายยอดขาย

### b. **Mean Absolute Error (MAE)**  
- **สูตร**: $ \text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i| $
- **ข้อดี**:
  - ทนทานต่อ Outliers มากกว่า MSE
- **ข้อเสีย**:
  - การคำนวณ Gradient อาจซับซ้อนกว่า MSE เล็กน้อย
- **การใช้งาน**:
  - เหมาะสำหรับงาน Regression ที่มี Outliers ในข้อมูล

### c. **Binary Cross-Entropy (BCE)**  
- **สูตร**: $ \text{BCE} = -\frac{1}{n} \sum_{i=1}^{n} [y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i)] $
- **ข้อดี**:
  - เหมาะสำหรับงาน Binary Classification
  - ช่วยให้โมเดลเรียนรู้ความแตกต่างระหว่างคลาสได้ดี
- **ข้อเสีย**:
  - ใช้ได้เฉพาะกรณีที่ค่า Output อยู่ในช่วง [0, 1]
- **การใช้งาน**:
  - เหมาะสำหรับงาน Binary Classification เช่น การตรวจจับ Spam Email, การจำแนกเพศ

### d. **Categorical Cross-Entropy (CCE)**  
- **สูตร**: $ \text{CCE} = -\sum_{i=1}^{n} y_i \log(\hat{y}_i) $
- **ข้อดี**:
  - เหมาะสำหรับงาน Multi-class Classification
  - ช่วยให้โมเดลเรียนรู้ความแตกต่างระหว่างหลายคลาสได้ดี
- **ข้อเสีย**:
  - ใช้ได้เฉพาะกรณีที่ค่า Output เป็น Probability Distribution (เช่น Softmax)
- **การใช้งาน**:
  - เหมาะสำหรับงาน Multi-class Classification เช่น การจำแนกรูปภาพ, การจำแนกประเภทข้อความ

### e. **Sparse Categorical Cross-Entropy**  
- **สูตร**: เหมือน CCE แต่ใช้ Label แบบ Integer (แทนที่จะเป็น One-Hot Encoding)
- **การใช้งาน**:
  - เหมาะสำหรับงาน Multi-class Classification เมื่อ Label เป็น Integer

### f. **Huber Loss**  
- **สูตร**:
  - กรณีที่ $ |y - \hat{y}| \leq \delta $:  
    $L_\delta(y, \hat{y}) = \frac{1}{2}(y - \hat{y})^2$
  - กรณีที่ $ |y - \hat{y}| > \delta $:  
    $L_\delta(y, \hat{y}) = \delta |y - \hat{y}| - \frac{1}{2}\delta^2$
- **ข้อดี**:
  - ผสมผสานระหว่าง MSE และ MAE โดยทนทานต่อ Outliers แต่ยังคงความแม่นยำในกรณีที่ค่าผิดพลาดเล็กน้อย
- **การใช้งาน**:
  - เหมาะสำหรับงาน Regression ที่ต้องการสมดุลระหว่าง MSE และ MAE

---

## 3. แนวทางในการเลือก Loss Function

### a. เลือกตามชนิดของปัญหา
- **Regression**:
  - Mean Squared Error (MSE): ใช้เมื่อข้อมูลไม่มี Outliers
  - Mean Absolute Error (MAE): ใช้เมื่อมี Outliers
  - Huber Loss: ใช้เมื่อต้องการสมดุลระหว่าง MSE และ MAE
- **Classification**:
  - Binary Cross-Entropy: ใช้สำหรับ Binary Classification
  - Categorical Cross-Entropy: ใช้สำหรับ Multi-class Classification
  - Sparse Categorical Cross-Entropy: ใช้สำหรับ Multi-class Classification ที่ Label เป็น Integer

### b. เลือกตามลักษณะของข้อมูล
- หากข้อมูลมี Outliers มาก → ใช้ MAE หรือ Huber Loss
