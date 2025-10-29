import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import io
import sys
import traceback

# Set page config FIRST - before any other Streamlit commands
st.set_page_config(
    page_title="Chest X-Ray Classifier",
    page_icon="🩺",
    layout="wide"
)

# Add error handling for the entire app
try:
    # Device configuration
    @st.cache_resource
    def get_device():
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    device = get_device()

    # Model architecture (must match training code)
    class ImprovedResNet50Classifier(nn.Module):
        def __init__(self, num_classes):
            super(ImprovedResNet50Classifier, self).__init__()
            self.resnet = models.resnet50(pretrained=False)
            
            # Improved classifier head with BatchNorm
            in_features = self.resnet.fc.in_features
            self.resnet.fc = nn.Sequential(
                nn.Linear(in_features, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes)
            )
            
            self.target_layer = self.resnet.layer4[-1]
        
        def forward(self, x):
            return self.resnet(x)

    # GradCAM implementation
    class GradCAM:
        def __init__(self, model, target_layer):
            self.model = model
            self.target_layer = target_layer
            self.gradients = None
            self.activations = None
            self.handles = []
            
            h1 = self.target_layer.register_forward_hook(self.save_activation)
            h2 = self.target_layer.register_full_backward_hook(self.save_gradient)
            self.handles = [h1, h2]
        
        def save_activation(self, module, input, output):
            self.activations = output.detach()
        
        def save_gradient(self, module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        def generate_cam(self, input_tensor, target_class=None):
            self.gradients = None
            self.activations = None
            
            input_tensor = input_tensor.clone()
            input_tensor.requires_grad = True
            
            self.model.eval()
            output = self.model(input_tensor)
            
            if target_class is None:
                target_class = output.argmax(dim=1).item()
            
            self.model.zero_grad()
            one_hot = torch.zeros_like(output)
            one_hot[0, target_class] = 1
            output.backward(gradient=one_hot, retain_graph=True)
            
            if self.gradients is None or self.activations is None:
                raise RuntimeError("Gradients or activations not captured")
            
            gradients = self.gradients[0]
            activations = self.activations[0]
            
            weights = gradients.mean(dim=(1, 2), keepdim=True)
            cam = (weights * activations).sum(dim=0)
            
            cam = torch.clamp(cam, min=0)
            cam = cam - cam.min()
            if cam.max() != 0:
                cam = cam / cam.max()
            
            cam = cam.unsqueeze(0).unsqueeze(0)
            cam = torch.nn.functional.interpolate(
                cam, size=(224, 224), mode='bilinear', align_corners=False
            )
            cam = cam.squeeze().cpu().numpy()
            
            return cam, target_class, output
        
        def remove_hooks(self):
            for handle in self.handles:
                handle.remove()

    def apply_colormap_on_image(org_img, activation_map, colormap=cv2.COLORMAP_JET):
        """Apply heatmap on image"""
        heatmap = cv2.resize(activation_map, (org_img.shape[1], org_img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, colormap)
        
        if len(org_img.shape) == 3 and org_img.shape[2] == 3:
            img_bgr = cv2.cvtColor(org_img, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = org_img
        
        superimposed_img = cv2.addWeighted(img_bgr, 0.6, heatmap, 0.4, 0)
        superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
        
        return superimposed_img

    @st.cache_resource
    def load_model(model_path):
        """Load the trained model"""
        try:
            checkpoint = torch.load(model_path, map_location=device)
            num_classes = checkpoint['num_classes']
            class_names = checkpoint['class_names']
            
            model = ImprovedResNet50Classifier(num_classes)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            model.eval()
            
            return model, class_names
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            st.error(traceback.format_exc())
            return None, None

    def preprocess_image(image):
        """Preprocess image for model input"""
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transform(image).unsqueeze(0)

    def predict_with_tta(model, image, device, num_augmentations=5):
        """Test-Time Augmentation for improved predictions"""
        model.eval()
        
        predictions = []
        tta_transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        with torch.no_grad():
            for _ in range(num_augmentations):
                input_tensor = tta_transforms(image).unsqueeze(0).to(device)
                output = model(input_tensor)
                predictions.append(torch.nn.functional.softmax(output, dim=1))
        
        avg_prediction = torch.stack(predictions).mean(dim=0)
        return avg_prediction

    def generate_gradcam(model, image, device):
        """Generate GradCAM visualization"""
        # Preprocess image
        img_array = np.array(image.resize((224, 224)))
        input_tensor = preprocess_image(image).to(device)
        
        # Create GradCAM
        gradcam = GradCAM(model, model.target_layer)
        
        try:
            cam, pred_class, output = gradcam.generate_cam(input_tensor)
            
            # Get probabilities
            probs = torch.nn.functional.softmax(output, dim=1)
            all_probs = probs[0].cpu().detach().numpy()
            
            # Apply colormap
            visualization = apply_colormap_on_image(img_array, cam)
            
            return cam, visualization, pred_class, all_probs
            
        finally:
            gradcam.remove_hooks()

    # Streamlit UI
    st.title("🩺 Chest X-Ray Classifier with GradCAM")
    st.markdown("---")

    # Display system info
    with st.expander("🔧 System Information"):
        st.write(f"**Device:** {device}")
        st.write(f"**PyTorch Version:** {torch.__version__}")
        st.write(f"**CUDA Available:** {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            st.write(f"**CUDA Version:** {torch.version.cuda}")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        model_path = st.text_input(
            "Model Path",
            value= r'C:\Users\HP\Downloads\resnet50_chest_xray_classifier_improved.pth',
            help="Path to the trained model file (.pth)"
        )
        
        # Check if file exists
        import os
        if os.path.exists(model_path):
            st.success(f"✓ Model file found")
        else:
            st.error(f"✗ Model file not found at: {model_path}")
            st.info("Please update the path to your model file")
        
        use_tta = st.checkbox(
            "Use Test-Time Augmentation",
            value=False,
            help="Improves accuracy but takes longer"
        )
        
        if use_tta:
            num_augmentations = st.slider(
                "Number of Augmentations",
                min_value=3,
                max_value=20,
                value=5,
                help="More augmentations = better accuracy but slower"
            )
        
        st.markdown("---")
        st.markdown("### About")
        st.info(
            """
            This app uses a ResNet50 model trained on chest X-ray images to classify:
            - Normal
            - Pneumonia
            - COVID-19
            - Tuberculosis
            
            Upload an X-ray image to get a prediction with GradCAM visualization.
            """
        )

    # Main content
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a chest X-ray image",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a chest X-ray image in PNG or JPG format"
        )
        
        if uploaded_file is not None:
            try:
                # Display uploaded image
                image = Image.open(uploaded_file).convert('RGB')
                st.image(image, caption="Uploaded X-Ray", use_container_width=True)
            except Exception as e:
                st.error(f"Error loading image: {str(e)}")
                image = None
        else:
            image = None

    with col2:
        st.header("🔍 Results")
        
        if uploaded_file is not None and image is not None:
            try:
                # Check if model file exists
                import os
                if not os.path.exists(model_path):
                    st.error(f"❌ Model file not found: {model_path}")
                    st.info("Please check the sidebar and update the model path.")
                else:
                    # Load model
                    with st.spinner("Loading model..."):
                        model, class_names = load_model(model_path)
                    
                    if model is None:
                        st.error("Failed to load model. Check the error message above.")
                    else:
                        # Make prediction
                        with st.spinner("Analyzing image..."):
                            if use_tta:
                                # TTA prediction
                                avg_prediction = predict_with_tta(model, image, device, num_augmentations)
                                pred_class = avg_prediction.argmax(dim=1).item()
                                all_probs = avg_prediction[0].cpu().detach().numpy()
                                confidence = all_probs[pred_class] * 100
                            else:
                                # Standard prediction
                                input_tensor = preprocess_image(image).to(device)
                                with torch.no_grad():
                                    output = model(input_tensor)
                                    probs = torch.nn.functional.softmax(output, dim=1)
                                    all_probs = probs[0].cpu().numpy()
                                    pred_class = all_probs.argmax()
                                    confidence = all_probs[pred_class] * 100
                        
                        # Display prediction
                        st.success("Analysis Complete!")
                        
                        # Prediction box
                        pred_label = class_names[pred_class]
                        
                        # Color based on confidence
                        if confidence > 80:
                            color = "green"
                        elif confidence > 60:
                            color = "orange"
                        else:
                            color = "red"
                        
                        st.markdown(f"""
                        <div style='padding: 20px; border-radius: 10px; background-color: #f0f2f6; text-align: center;'>
                            <h2 style='color: {color}; margin: 0;'>{pred_label}</h2>
                            <h3 style='color: #555; margin: 10px 0 0 0;'>Confidence: {confidence:.2f}%</h3>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown("---")
                        
                        # Probability distribution
                        st.subheader("📊 Class Probabilities")
                        prob_data = {class_names[i]: all_probs[i] * 100 for i in range(len(class_names))}
                        
                        # Sort by probability
                        sorted_probs = sorted(prob_data.items(), key=lambda x: x[1], reverse=True)
                        
                        for class_name, prob in sorted_probs:
                            st.progress(float(prob / 100))
                            st.text(f"{class_name}: {prob:.2f}%")
                        
                        # Store results for GradCAM section
                        st.session_state['prediction_made'] = True
                        st.session_state['model'] = model
                        st.session_state['class_names'] = class_names
                        st.session_state['image'] = image
                        
            except Exception as e:
                st.error(f"❌ Error during prediction: {str(e)}")
                st.code(traceback.format_exc())
        else:
            st.info("👆 Please upload an X-ray image to get started")

    # GradCAM Visualization Section
    if uploaded_file is not None and image is not None and 'prediction_made' in st.session_state and st.session_state['prediction_made']:
        st.markdown("---")
        st.header("🔥 GradCAM Visualization")
        st.markdown("*Heatmap showing regions that influenced the model's decision*")
        
        try:
            with st.spinner("Generating GradCAM..."):
                model = st.session_state['model']
                class_names = st.session_state['class_names']
                image = st.session_state['image']
                
                cam, visualization, pred_class, all_probs = generate_gradcam(model, image, device)
                confidence = all_probs[pred_class] * 100
            
            # Create three columns for visualization
            viz_col1, viz_col2, viz_col3 = st.columns(3)
            
            with viz_col1:
                st.subheader("Original Image")
                st.image(image.resize((224, 224)), use_container_width=True)
            
            with viz_col2:
                st.subheader("Heatmap")
                fig, ax = plt.subplots(figsize=(6, 6))
                im = ax.imshow(cam, cmap='jet')
                ax.axis('off')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                st.pyplot(fig)
                plt.close()
            
            with viz_col3:
                st.subheader("Overlay")
                st.image(visualization, use_container_width=True)
            
            # Explanation
            st.markdown("""
            ### 📖 Understanding the Heatmap
            
            - **Red/Orange areas**: Regions that most strongly influenced the prediction
            - **Blue areas**: Regions with minimal influence
            - The overlay combines the original image with the heatmap to show exactly where the model is "looking"
            
            This visualization helps understand whether the model is focusing on clinically relevant regions.
            """)
            
            # Download button for visualization
            st.markdown("---")
            
            # Create downloadable image
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            axes[0].imshow(image.resize((224, 224)))
            axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
            axes[0].axis('off')
            
            im = axes[1].imshow(cam, cmap='jet')
            axes[1].set_title('GradCAM Heatmap', fontsize=12, fontweight='bold')
            axes[1].axis('off')
            plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
            
            axes[2].imshow(visualization)
            axes[2].set_title(f'Prediction: {class_names[pred_class]}\nConfidence: {confidence:.2f}%', 
                             fontsize=12, fontweight='bold')
            axes[2].axis('off')
            
            plt.tight_layout()
            
            # Save to buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            plt.close()
            
            st.download_button(
                label="📥 Download Visualization",
                data=buf,
                file_name=f"gradcam_{class_names[pred_class]}.png",
                mime="image/png"
            )
            
        except Exception as e:
            st.error(f"Error generating GradCAM: {str(e)}")
            st.code(traceback.format_exc())

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #888;'>
        <p>⚠️ <strong>Disclaimer:</strong> This tool is for educational purposes only and should not be used for medical diagnosis. 
        Always consult with qualified healthcare professionals for medical advice.</p>
    </div>
    """, unsafe_allow_html=True)

except Exception as e:
    st.error("### ❌ Critical Error")
    st.error(f"The application encountered a critical error: {str(e)}")
    st.code(traceback.format_exc())
    st.info("Please check your Python environment and ensure all dependencies are installed correctly.")