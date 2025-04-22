import torch
import cv2 as cv
import tkinter as tk
from tkinter import filedialog, messagebox
from models import Z_DCE_Network, SR_CNN
import numpy as np
import os
from PIL import Image, ImageTk

class Processor:
    def __init__(self, z_dce_model_path, sr_cnn_model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.z_dce_network = Z_DCE_Network().to(self.device)
        self.sr_cnn = SR_CNN().to(self.device)
        self.z_dce_network.load_state_dict(torch.load(z_dce_model_path, map_location=self.device))
        self.sr_cnn.load_state_dict(torch.load(sr_cnn_model_path, map_location=self.device))
        self.z_dce_network.eval()
        self.sr_cnn.eval()

    def process_image(self, image):
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).float() / 255.0
        image_tensor = image_tensor.to(self.device)

        img_gray = torch.mean(image_tensor, dim=1, keepdim=True)
        z = self.z_dce_network(img_gray)
        z = z.repeat(1, 3, 1, 1)
        sr = self.sr_cnn(z)

        intensity = torch.mean(image_tensor, dim=1, keepdim=True)
        sr_temp = sr * 0.1 + image_tensor * 0.8

        if np.mean(sr_temp.cpu().detach().numpy()) < 0.5:
            sr = sr_temp
        else:
            sr = sr * 0.15 + image_tensor * 0.8

        sr = torch.clamp(sr, 0, 1).squeeze().permute(1, 2, 0).cpu().detach().numpy()
        sr = (sr * 255).astype(np.uint8)
        return sr

    def process_video(self, input_path, output_path, update_callback):
        cap = cv.VideoCapture(input_path)
        fourcc = cv.VideoWriter_fourcc(*'XVID')
        width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv.CAP_PROP_FPS))

        out = cv.VideoWriter(output_path, fourcc, fps, (width, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame = self.process_image(frame)
            out.write(processed_frame)

            update_callback(frame, processed_frame)

        cap.release()
        out.release()

class GUI:
    def __init__(self):
        self.processor = Processor("z_dce_network_ZESR.pth", "sr_cnn_ZESR.pth")
        self.root = tk.Tk()
        self.root.title("Zero-DCE Super Resolution Testing")
        self.root.attributes("-fullscreen", True)
        self.root.bind("<Escape>", self.exit_fullscreen) 
        self.setup_ui()
    def setup_ui(self):
        image_frame = tk.Frame(self.root)
        image_frame.pack(pady=10)
        tk.Label(image_frame, text="Original Image").grid(row=0, column=0, padx=10)
        tk.Label(image_frame, text="Processed Image").grid(row=0, column=1, padx=10)
        self.original_image = tk.Label(image_frame)
        self.original_image.grid(row=1, column=0, padx=10)
        self.processed_image = tk.Label(image_frame)
        self.processed_image.grid(row=1, column=1, padx=10)
        tk.Button(self.root, text="Select Image", command=self.process_image).pack(pady=10)
        tk.Button(self.root, text="Select Video", command=self.process_video).pack(pady=10)
        tk.Button(self.root, text="Exit", command=self.root.quit).pack(pady=10)

    def update_images(self, original_frame, processed_frame):
        original_frame_rgb = cv.cvtColor(original_frame, cv.COLOR_BGR2RGB)
        original_img = Image.fromarray(original_frame_rgb)
        original_img.thumbnail((800, 800))
        original_img_tk = ImageTk.PhotoImage(image=original_img)
        processed_frame_rgb = cv.cvtColor(processed_frame, cv.COLOR_BGR2RGB)
        processed_img = Image.fromarray(processed_frame_rgb)
        processed_img.thumbnail((800, 800))
        processed_img_tk = ImageTk.PhotoImage(image=processed_img)
        self.original_image.img_tk = original_img_tk
        self.processed_image.img_tk = processed_img_tk
        self.original_image.configure(image=original_img_tk)
        self.processed_image.configure(image=processed_img_tk)
        self.root.update()


    def process_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
        if not file_path:
            return

        image = cv.imread(file_path)
        processed_image = self.processor.process_image(image)
        self.update_images(image, processed_image)
        #saving processed image with location and name
        save_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG Files", "*.png")])
        if save_path:
            cv.imwrite(save_path, processed_image)
            messagebox.showinfo("Success", f"Processed image saved to {save_path}")

    def process_video(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi;*.mov")])
        if not file_path:
            return
        #saving processed video with location and name
        save_path = filedialog.asksaveasfilename(defaultextension=".avi", filetypes=[("AVI Files", "*.avi")])
        if save_path:
            self.processor.process_video(file_path, save_path, self.update_images)
            messagebox.showinfo("Success", f"Processed video saved to {save_path}")

    def run(self):
        self.root.mainloop()
    def exit_fullscreen(self, event=None):
        self.root.attributes("-fullscreen", False)


if __name__ == "__main__":
    gui = GUI()
    gui.run()
