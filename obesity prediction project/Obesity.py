import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import joblib
import pickle
import customtkinter as ctk  # For modern UI components
from PIL import Image, ImageTk  # For handling images
import pandas as pd
from tkinter import filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Set the appearance mode and default color theme
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

# Set this to True to enable the debug tab
DEBUG_MODE = True

# Encodings from the first paste - keeping original logic
gender_map = {"Male": 1, "Female": 0}
yes_no_map = {"yes": 1, "no": 0}
caec_map = {"Never": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}
calc_map = {"Never": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}
mtrans_map = {"Automobile": 0, "Bike": 1, "Motorbike": 2, "Public_Transportation": 3, "Walking": 4}

# Mapping for obesity categories (from first paste)
ordinal_mapping = {
    'Insufficient_Weight': 0,
    'Normal_Weight': 1,
    'Overweight_Level_I': 2,
    'Overweight_Level_II': 3,
    'Obesity_Type_I': 4,
    'Obesity_Type_II': 5,
    'Obesity_Type_III': 6
}
reverse_mapping = {v: k for k, v in ordinal_mapping.items()}

# Load trained model
try:
    model = joblib.load('obesity_model.pkl')
except FileNotFoundError:
    model = None

# Try to load scaler
try:
    scaler = joblib.load('obesity_scaler.pkl')
except:
    # Create a placeholder scaler if one doesn't exist
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()

# Standard BMI classification function (from first paste)
def classify_by_bmi(bmi):
    if bmi < 18.5:
        return "Insufficient_Weight"
    elif bmi < 25:
        return "Normal_Weight"
    elif bmi < 27.5:
        return "Overweight_Level_I"
    elif bmi < 30:
        return "Overweight_Level_II"
    elif bmi < 35:
        return "Obesity_Type_I"
    elif bmi < 40:
        return "Obesity_Type_II"
    else:
        return "Obesity_Type_III"

# Create an ensemble prediction system that combines model prediction with BMI-based rules (from first paste)
def ensemble_predict(features, bmi):
    try:
        # Get model prediction
        model_prediction = model.predict(features)[0]
        model_label = reverse_mapping.get(model_prediction, "Unknown")
        
        # Get BMI-based prediction
        bmi_prediction = classify_by_bmi(bmi)
        
        # Confidence check - if model prediction seems far off from BMI expectation,
        # favor the BMI prediction or a weighted decision
        model_ordinal = ordinal_mapping.get(model_label, 0)
        bmi_ordinal = ordinal_mapping.get(bmi_prediction, 0)
        
        # If predictions are very different (more than 2 categories apart)
        if abs(model_ordinal - bmi_ordinal) > 2:
            # Favor BMI prediction but adjust slightly toward model prediction
            final_ordinal = bmi_ordinal + (model_ordinal - bmi_ordinal) // 3
            final_label = reverse_mapping.get(final_ordinal, bmi_prediction)
            return final_label, model_label, bmi_prediction, "ensemble (bmi-weighted)"
            
        # If predictions are slightly different (1-2 categories apart)
        elif abs(model_ordinal - bmi_ordinal) > 0:
            # For slight differences, trust the model but with BMI as a sanity check
            # Use weighted average favoring the model slightly
            final_ordinal = round((model_ordinal * 0.6) + (bmi_ordinal * 0.4))
            final_label = reverse_mapping.get(final_ordinal, model_label)
            return final_label, model_label, bmi_prediction, "ensemble (model-weighted)"
            
        # If predictions match
        else:
            return model_label, model_label, bmi_prediction, "both agree"
            
    except Exception as e:
        # Fall back to BMI prediction if model fails
        bmi_prediction = classify_by_bmi(bmi)
        return bmi_prediction, "error", bmi_prediction, "BMI fallback"

class ObesityPredictionApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        # Configure main window
        self.title("Obesity Risk Assessment System")
        self.geometry("1000x850")
        self.configure(fg_color="#212121")
        
        # Create main frame with padding
        self.main_frame = ctk.CTkFrame(self, corner_radius=15, fg_color="#3A3A3A", border_width=1, border_color="#B6B6FF")
        self.main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Create header
        self.create_header()
        
        # Create a notebook for organizing inputs
        self.create_notebook()
        
        # Create result and prediction section
        self.create_result_section()

    def create_header(self):
        """Create the header with logo and title"""
        header_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        header_frame.pack(fill="x", padx=20, pady=(20, 10))
        
        # Title and subtitle
        title_frame = ctk.CTkFrame(header_frame, fg_color="transparent")
        title_frame.pack(side="left")
        
        ctk.CTkLabel(title_frame, text="Obesity Risk Assessment", 
                     font=ctk.CTkFont(family="Arial", size=24, weight="bold"),
                     text_color="#B6B6FF").pack(anchor="w")
        
        ctk.CTkLabel(title_frame, text="Enter your information to calculate obesity risk category",
                     font=ctk.CTkFont(family="Arial", size=14),
                     text_color="#E0E0FF").pack(anchor="w")
        
    
    def create_notebook(self):
        """Create tabbed interface for inputs"""
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(fill="both", expand=True, padx=20, pady=10)


        style = ttk.Style()
        style.theme_use('default')
        style.configure("TNotebook", background="#333333", borderwidth=1)
        style.configure("TNotebook.Tab", padding=[10, 5], font=('Arial', 11), background="#d9d9d9")
        style.map("TNotebook.Tab",
                  background=[("selected", "#292929"),("!active", "#333333")],
                  foreground=[("selected", "#B6B6FF"),("!active", "#E0E0FF")])


        tab_bg_color = "#333333"

        self.personal_tab = ctk.CTkFrame(self.notebook, fg_color=tab_bg_color)
        self.dietary_tab = ctk.CTkFrame(self.notebook, fg_color=tab_bg_color)
        self.lifestyle_tab = ctk.CTkFrame(self.notebook, fg_color=tab_bg_color)

        self.notebook.add(self.personal_tab, text="Personal Information")
        self.notebook.add(self.dietary_tab, text="Dietary Habits")
        self.notebook.add(self.lifestyle_tab, text="Lifestyle Factors")


        if DEBUG_MODE:
            self.debug_tab = ctk.CTkFrame(self.notebook, fg_color=tab_bg_color)
            self.notebook.add(self.debug_tab, text="Debug")
            self.create_debug_tab()


        self.create_personal_tab()
        self.create_dietary_tab()
        self.create_lifestyle_tab()

    
    def create_personal_tab(self):
        """Create inputs for personal information"""
        # Create two columns for better layout
        left_frame = ctk.CTkFrame(self.personal_tab, fg_color="transparent")
        left_frame.pack(side="left", fill="both", expand=True, padx=(20, 10), pady=20)
        
        right_frame = ctk.CTkFrame(self.personal_tab, fg_color="transparent")
        right_frame.pack(side="right", fill="both", expand=True, padx=(10, 20), pady=20)
        
        # Left column inputs
        self.gender_var = tk.StringVar(value="Male")
        self.create_dropdown(left_frame, "Gender", self.gender_var, ["Male", "Female"], 0)
        
        self.create_entry(left_frame, "Age", 1)
        self.age_entry = self.entry  # Store reference
        
        self.create_entry(left_frame, "Height (m)", 2)
        self.height_entry = self.entry  # Store reference
        
        # Right column inputs
        self.create_entry(right_frame, "Weight (kg)", 0)
        self.weight_entry = self.entry  # Store reference
        
        self.fhwo_var = tk.StringVar(value="no")
        self.create_dropdown(right_frame, "Family History of Overweight", self.fhwo_var, ["yes", "no"], 1)
        
        # BMI Display (calculated)
        bmi_frame = ctk.CTkFrame(right_frame, fg_color="#212121", corner_radius=10)
        bmi_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(20, 0), padx=5)
        
        ctk.CTkLabel(bmi_frame, text="BMI Calculator", 
                    font=ctk.CTkFont(family="Arial", size=14, weight="bold"),
                    text_color="#B6B6FF").pack(pady=(10, 5))
        
        self.bmi_value = ctk.CTkLabel(bmi_frame, text="--.-", 
                                    font=ctk.CTkFont(family="Arial", size=24, weight="bold"),
                                    text_color="#E0E0FF")
        self.bmi_value.pack()
        
        self.bmi_category = ctk.CTkLabel(bmi_frame, text="Enter height and weight", 
                                        text_color="#E0E0FF")
        self.bmi_category.pack(pady=(0, 10))
        
        # Calculate BMI button
        ctk.CTkButton(bmi_frame, text="Calculate BMI", 
                     command=self.calculate_bmi,
                     font=ctk.CTkFont(family="Arial", size=12),
                     fg_color="#B6B6FF", hover_color="#A3A3E0", text_color="#333333",
                     corner_radius=6).pack(pady=(0, 10))

    def create_dietary_tab(self):
        """Create inputs for dietary information"""
        # Create two columns
        left_frame = ctk.CTkFrame(self.dietary_tab, fg_color="transparent")
        left_frame.pack(side="left", fill="both", expand=True, padx=(20, 10), pady=20)
        
        right_frame = ctk.CTkFrame(self.dietary_tab, fg_color="transparent")
        right_frame.pack(side="right", fill="both", expand=True, padx=(10, 20), pady=20)
        
        # Left column inputs
        self.favc_var = tk.StringVar(value="no")
        self.create_dropdown(left_frame, "High Calorie Food Consumption (FAVC)", self.favc_var, ["yes", "no"], 0)
        
        self.fcvc_var = tk.StringVar(value="2")
        self.create_dropdown(left_frame, "Vegetable Consumption (FCVC)", self.fcvc_var, ["1", "2", "3"], 1,
                          description="1 = Low, 2 = Medium, 3 = High")
        
        # NCP
        self.create_entry(left_frame, "Number of Meals per Day (NCP)", 2)
        self.ncp_entry = self.entry  # Store reference
        self.ncp_entry.insert(0, "3")  # Default value
        
        # Right column inputs
        self.caec_var = tk.StringVar(value="Sometimes")
        self.create_dropdown(right_frame, "Food Between Meals (CAEC)", self.caec_var, 
                          ["Never", "Sometimes", "Frequently", "Always"], 0)
        
        self.ch2o_var = tk.StringVar(value="2")
        self.create_dropdown(right_frame, "Daily Water Consumption (CH2O)", self.ch2o_var, ["1", "2", "3"], 1,
                          description="1 = Less than 1L, 2 = 1-2L, 3 = More than 2L")
        
        self.scc_var = tk.StringVar(value="no")
        self.create_dropdown(right_frame, "Monitor Calorie Intake (SCC)", self.scc_var, ["yes", "no"], 2)
        
    def create_lifestyle_tab(self):
        """Create inputs for lifestyle information"""
        # Create two columns
        left_frame = ctk.CTkFrame(self.lifestyle_tab, fg_color="transparent")
        left_frame.pack(side="left", fill="both", expand=True, padx=(20, 10), pady=20)
        
        right_frame = ctk.CTkFrame(self.lifestyle_tab, fg_color="transparent")
        right_frame.pack(side="right", fill="both", expand=True, padx=(10, 20), pady=20)
        
        # Left column inputs
        self.faf_var = tk.StringVar(value="1")
        self.create_dropdown(left_frame, "Physical Activity Frequency (FAF)", self.faf_var, ["0", "1", "2", "3"], 0,
                          description="0 = None, 1 = Low, 2 = Medium, 3 = High")
        
        self.tue_var = tk.StringVar(value="1")
        self.create_dropdown(left_frame, "Technology Usage Time (TUE)", self.tue_var, ["0", "1", "2", "3"], 1,
                          description="0 = 0-2 hours, 1 = 3-5 hours, 2 = 6-8 hours, 3 = >8 hours")
        
        self.smoke_var = tk.StringVar(value="no")
        self.create_dropdown(left_frame, "Do You Smoke?", self.smoke_var, ["yes", "no"], 2)

        self.calc_var = tk.StringVar(value="Sometimes")
        self.create_dropdown(right_frame, "Alcohol Consumption (CALC)", self.calc_var, 
                  ["Never", "Sometimes", "Frequently", "Always"], 1)

        self.mtrans_var = tk.StringVar(value="Public Transportation")
        self.create_dropdown(right_frame, "Primary Transportation Mode (MTRANS)", self.mtrans_var, 
                  ["Automobile", "Bike", "Motorbike", "Public Transportation", "Walking"], 2)
        
        # Add information panel
        info_frame = ctk.CTkFrame(right_frame, fg_color="#212121", corner_radius=10)
        info_frame.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(20, 0), padx=5)
        
        ctk.CTkLabel(info_frame, text="ℹ Lifestyle Impact", 
                    font=ctk.CTkFont(family="Arial", size=14, weight="bold"),
                    text_color="#B6B6FF").pack(pady=(10, 5))
        
        ctk.CTkLabel(info_frame, text="Regular physical activity and reducing\nsedentary time can significantly decrease\nobesity risk and improve overall health.",
                    text_color="#E0E0FF",
                    justify="center").pack(pady=(0, 20), padx=(10, 10))

    def create_result_section(self):
        """Create result display and prediction button"""
        # Result frame
        result_frame = ctk.CTkFrame(self.main_frame, fg_color="#333333", corner_radius=15, 
                                 border_width=1, border_color="#B6B6FF")
        result_frame.pack(fill="x", padx=20, pady=(0, 20))
        
        # Prediction button
        predict_button = ctk.CTkButton(result_frame, text="Assess Obesity Risk", 
                     command=self.predict,
                     font=ctk.CTkFont(family="Arial", size=16, weight="bold"),
                     height=50, fg_color="#B6B6FF", hover_color="#8E8ECC",
                     corner_radius=6, text_color="#333333")
        predict_button.pack(padx=20, pady=20)
        
        # If model is not loaded, disable the button
        if model is None:
            predict_button.configure(state="disabled", fg_color="#bbbbbb", 
                                    text="Model Not Available")
        
        # Result display
        self.result_label = ctk.CTkLabel(result_frame, text="Complete all fields and click the button to assess your obesity risk",
                                       font=ctk.CTkFont(family="Arial", size=14),
                                       text_color="#E0E0FF")
        self.result_label.pack(pady=(0, 10))
        
        # Details for prediction results
        self.result_details = ctk.CTkLabel(result_frame, text="",
                                        font=ctk.CTkFont(family="Arial", size=12),
                                        text_color="#E0E0FF")
        self.result_details.pack(pady=(0, 20))
        
        # Add disclaimer
        ctk.CTkLabel(self.main_frame, text="Disclaimer: This tool provides an estimation only and is not a medical diagnosis. Consult healthcare professionals for proper assessment.",
                    font=ctk.CTkFont(size=11),
                    text_color="#777777").pack(padx=20, pady=(0, 10))

    def create_dropdown(self, parent, label_text, variable, options, row, description=None):
        """Helper to create a styled dropdown with optional description"""
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.grid(row=row, column=0, sticky="ew", pady=10, padx=5)
        
        # Label
        ctk.CTkLabel(frame, text=label_text,
                    font=ctk.CTkFont(family="Arial", size=12),
                    text_color="#E0E0FF").pack(anchor="w")
        
        # Optional description
        if description:
            ctk.CTkLabel(frame, text=description,
                        font=ctk.CTkFont(family="Arial", size=10),
                        text_color="#777777").pack(anchor="w")
        
        # Dropdown (ComboBox)
        dropdown = ctk.CTkComboBox(frame, values=options, variable=variable,
                                 width=200, border_color="#B6B6FF",
                                 button_color="#B6B6FF", button_hover_color="#8E8ECC",
                                 dropdown_fg_color="#333333", dropdown_hover_color="#222222",
                                 dropdown_text_color="#FDFDFD", fg_color="#333333", text_color="#FDFDFD")
        dropdown.pack(anchor="w", pady=(5, 0))

    def create_entry(self, parent, label_text, row):
        """Helper to create a styled entry field"""
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.grid(row=row, column=0, sticky="ew", pady=10, padx=5)
        
        # Label
        ctk.CTkLabel(frame, text=label_text,
                    font=ctk.CTkFont(family="Arial", size=12),
                    text_color="#E0E0FF").pack(anchor="w")
        
        # Entry
        self.entry = ctk.CTkEntry(frame, width=200, border_color="#B6B6FF", fg_color="#333333", text_color="#E0E0FF")
        self.entry.pack(anchor="w", pady=(5, 0))

    def calculate_bmi(self):
        """Calculate BMI from height and weight"""
        try:
            height = float(self.height_entry.get())
            weight = float(self.weight_entry.get())
            
            if height <= 0 or weight <= 0:
                self.bmi_value.configure(text="Invalid")
                self.bmi_category.configure(text="Enter valid height and weight")
                return None
                
            bmi = weight / (height * height)
            self.bmi_value.configure(text=f"{bmi:.1f}")
            
            # Determine BMI category based on the first paste logic
            bmi_category = classify_by_bmi(bmi)
            
            # Set display category for UI
            display_category = bmi_category.replace("_", " ")
            
            # Set color based on category
            if bmi_category == "Insufficient_Weight":
                color = "#ffa500"  # Orange for underweight
            elif bmi_category == "Normal_Weight":
                color = "#4caf50"  # Green for normal
            elif bmi_category.startswith("Overweight"):
                color = "#ff9800"  # Orange for overweight
            else:  # Obesity
                color = "#f44336"  # Red for obese
                
            self.bmi_category.configure(text=display_category, text_color=color)
            return bmi
            
        except ValueError:
            self.bmi_value.configure(text="Error")
            self.bmi_category.configure(text="Enter valid numbers", text_color="#f44336")
            return None

    def validate_inputs(self):
        """Validate all inputs before prediction"""
        try:
            # Check required numerical fields
            if not self.age_entry.get().strip():
                return False, "Age is required"
            float(self.age_entry.get())
            
            if not self.height_entry.get().strip():
                return False, "Height is required"
            float(self.height_entry.get())
            
            if not self.weight_entry.get().strip():
                return False, "Weight is required"
            float(self.weight_entry.get())
            
            if not self.ncp_entry.get().strip():
                return False, "Number of meals is required"
            float(self.ncp_entry.get())
            
            # All validations passed
            return True, ""
        except ValueError:
            return False, "Please enter valid numbers"

    def predict(self):
        """Predict obesity category based on inputs using the loaded model"""
        # Validate inputs
        valid, message = self.validate_inputs()
        if not valid:
            self.result_label.configure(text=message, text_color="#f44336")
            return
            
        try:
            # Calculate BMI first to use in prediction
            bmi = self.calculate_bmi()
            if bmi is None:
                self.result_label.configure(text="Error calculating BMI", text_color="#f44336")
                return
                
            # Extract values from the UI
            gender = self.gender_var.get()
            age = float(self.age_entry.get())
            height = float(self.height_entry.get())
            weight = float(self.weight_entry.get())
            fhwo = self.fhwo_var.get()
            favc = self.favc_var.get()
            fcvc = float(self.fcvc_var.get())
            ncp = float(self.ncp_entry.get())
            caec = self.caec_var.get()
            ch2o = float(self.ch2o_var.get())
            scc = self.scc_var.get()
            faf = float(self.faf_var.get())
            tue = float(self.tue_var.get())
            
            # Map categorical inputs (using first paste encoding)
            gender_numeric = gender_map.get(gender, 0)
            fhwo_numeric = yes_no_map.get(fhwo, 0)
            favc_numeric = yes_no_map.get(favc, 0)
            caec_numeric = caec_map.get(caec, 0)
            scc_numeric = yes_no_map.get(scc, 0)
            
            # Create feature vector (order from first paste)
            features_raw = np.array([[
                gender_numeric,    # Gender
                age,               # Age
                height,            # Height
                weight,            # Weight
                fhwo_numeric,      # Family history with overweight
                favc_numeric,      # FAVC (high calorie food)
                fcvc,              # FCVC (vegetable consumption)
                ncp,               # NCP (meals per day)
                caec_numeric,      # CAEC (food between meals)
                ch2o,              # CH2O (water intake)
                scc_numeric,       # SCC (calorie monitoring)
                faf,               # FAF (physical activity)
                tue,               # TUE (technology usage)
                bmi                # BMI (calculated)
            ]])
            
            # Normalize features if we have a scaler
            features = features_raw
            if scaler is not None and hasattr(scaler, 'mean_'):
                features = scaler.transform(features_raw)
            
            # Use the ensemble prediction system from first paste
            if model is not None:
                final_prediction, model_prediction, bmi_prediction, method = ensemble_predict(features, bmi)
            else:
                # Fall back to BMI only if model is not available
                bmi_prediction = classify_by_bmi(bmi)
                final_prediction, model_prediction, _, method = bmi_prediction, "N/A", bmi_prediction, "BMI only (no model)"
            
            # Display the result with styling
            result_text = final_prediction.replace("_", " ")
            self.result_label.configure(
                text=f"Predicted Obesity Category: {result_text}",
                font=ctk.CTkFont(family="Arial", size=16, weight="bold"),
                text_color="#B6B6FF"
            )
            
            # Show method and details
            details_text = f"Method: {method}"
            if model_prediction != "N/A" and model_prediction != "error":
                details_text += f" | Model prediction: {model_prediction.replace('_', ' ')}"
            if bmi_prediction:
                details_text += f" | BMI prediction: {bmi_prediction.replace('_', ' ')}"
                
            self.result_details.configure(text=details_text)

            if DEBUG_MODE:
                self.log_message(f"Prediction: {final_prediction} | Method: {method}")
                self.log_message(f"Input features: Gender={gender}, Age={age}, BMI={bmi:.2f}")
            
            
        except Exception as e:
            self.result_label.configure(
                text=f"Error during prediction: {str(e)}",
                text_color="#f44336"
            )
            if DEBUG_MODE:
                self.log_message(f"ERROR: Prediction failed - {str(e)}")
    
    def create_debug_tab(self):
        """Create the debug tab with dataset examples and debugging options"""
        # Create main frame for the debug tab
        debug_frame = ctk.CTkFrame(self.debug_tab, fg_color="#ffffff")
        debug_frame.pack(fill="both", expand=True, padx=10, pady=10)
    
        # Create tabs within debug tab
        debug_notebook = ttk.Notebook(debug_frame)
        debug_notebook.pack(fill="both", expand=True, padx=5, pady=5)
    
        # Dataset examples tab
        dataset_frame = ctk.CTkFrame(debug_notebook, fg_color="#ffffff")
        debug_notebook.add(dataset_frame, text="Dataset Examples")
    
        # Create treeview to display dataset examples
        columns = ["Gender", "Age", "Height", "Weight", "FHWO", "FAVC", "FCVC", "NCP", 
                   "CAEC", "SMOKE", "CH2O", "SCC", "FAF", "TUE", "CALC", "MTRANS", "NObeyesdad"]
    
        tree = ttk.Treeview(dataset_frame, columns=columns, show="headings", height=10)
    
        # Configure column widths and headings
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=70, anchor="center")
    
        tree.pack(fill="both", expand=True, padx=5, pady=5)
    
        # Add scrollbar
        scrollbar = ttk.Scrollbar(dataset_frame, orient="vertical", command=tree.yview)
        scrollbar.pack(side="right", fill="y")
        tree.configure(yscrollcommand=scrollbar.set)
    
        # Button to load dataset
        load_frame = ctk.CTkFrame(dataset_frame, fg_color="transparent")
        load_frame.pack(fill="x", padx=5, pady=5)
    
        ctk.CTkButton(load_frame, text="Load Dataset", 
                     command=lambda: self.load_dataset(tree),
                     font=ctk.CTkFont(family="Arial", size=12),
                     fg_color="#B6B6FF", hover_color="#B6B6FF",
                     corner_radius=6).pack(side="left", padx=5)
    
        ctk.CTkButton(load_frame, text="Load Example", 
                     command=lambda: self.load_example_from_selection(tree),
                     font=ctk.CTkFont(family="Arial", size=12),
                     fg_color="#4caf50", hover_color="#3d8b40",
                     corner_radius=6).pack(side="left", padx=5)
    
        # Model testing tab
        testing_frame = ctk.CTkFrame(debug_notebook, fg_color="#ffffff")
        debug_notebook.add(testing_frame, text="Model Testing")
    
        # Add model info section
        info_frame = ctk.CTkFrame(testing_frame, fg_color="#f0f7ff", corner_radius=10)
        info_frame.pack(fill="x", padx=10, pady=10)

        ctk.CTkLabel(info_frame, text="Model Information", 
                     font=ctk.CTkFont(family="Arial", size=14, weight="bold"),
                     text_color="#B6B6FF").pack(pady=(10, 5))
    
        if model is not None:
            model_type = type(model).__name__
            model_info_text = f"Model Type: {model_type}"
        else:
            model_info_text = "No model loaded"
    
        ctk.CTkLabel(info_frame, text=model_info_text, 
                    text_color="#333333").pack(pady=(0, 5))
    
        # Add prediction visualization
        self.viz_frame = ctk.CTkFrame(testing_frame, fg_color="#ffffff", corner_radius=10)
        self.viz_frame.pack(fill="both", expand=True, padx=10, pady=10)
    
        ctk.CTkButton(testing_frame, text="Run Batch Predictions", 
                     command=self.run_batch_predictions,
                     font=ctk.CTkFont(family="Arial", size=12),
                     fg_color="#ff9800", hover_color="#f57c00",
                     corner_radius=6).pack(pady=10)
    
        # Logs tab
        logs_frame = ctk.CTkFrame(debug_notebook, fg_color="#ffffff")
        debug_notebook.add(logs_frame, text="Logs")
    
        # Add text widget for logs
        self.logs_text = ctk.CTkTextbox(logs_frame, width=400, height=300, corner_radius=0)
        self.logs_text.pack(fill="both", expand=True, padx=5, pady=5)
    
        # Add log control buttons
        log_buttons_frame = ctk.CTkFrame(logs_frame, fg_color="transparent")
        log_buttons_frame.pack(fill="x", padx=5, pady=5)
    
        ctk.CTkButton(log_buttons_frame, text="Clear Logs", 
                     command=lambda: self.logs_text.delete(1.0, "end"),
                     font=ctk.CTkFont(family="Arial", size=12),
                     fg_color="#f44336", hover_color="#d32f2f",
                     corner_radius=6).pack(side="left", padx=5)
    
        ctk.CTkButton(log_buttons_frame, text="Save Logs", 
                     command=self.save_logs,
                     font=ctk.CTkFont(family="Arial", size=12),
                     fg_color="#B6B6FF", hover_color="#B6B6FF",
                     corner_radius=6).pack(side="left", padx=5)


    def load_dataset(self, tree):
        """Load dataset from CSV file and display in treeview"""
        try:
            # Open file dialog
            file_path = filedialog.askopenfilename(
                title="Select Dataset CSV File",
                filetypes=[("CSV files", "*.csv")]
            )
        
            if not file_path:
                return
            
            # Load dataset
            dataset = pd.read_csv(file_path)
        
            # Clear existing data
            for item in tree.get_children():
                tree.delete(item)
            
            # Add data to treeview
            for _, row in dataset.iterrows():
                values = [row[col] if col in row else "" for col in [
                    "Gender", "Age", "Height", "Weight", "family_history_with_overweight", 
                    "FAVC", "FCVC", "NCP", "CAEC", "SMOKE", "CH2O", "SCC", "FAF", 
                    "TUE", "CALC", "MTRANS", "NObeyesdad"
                ]]
                tree.insert("", "end", values=values)
            
            # Log action
            self.log_message(f"Loaded dataset from {file_path} with {len(dataset)} rows")
        
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load dataset: {str(e)}")
            self.log_message(f"ERROR: Failed to load dataset - {str(e)}")


    def load_example_from_selection(self, tree):
        """Load selected example data into the form"""
        selected_item = tree.selection()
        if not selected_item:
            messagebox.showinfo("Info", "Please select an example from the dataset")
            return
        
        # Get values from selected row
        values = tree.item(selected_item[0], "values")
    
        # Set values in form
        self.gender_var.set(values[0])
        self.age_entry.delete(0, "end")
        self.age_entry.insert(0, values[1])
        self.height_entry.delete(0, "end")
        self.height_entry.insert(0, values[2])
        self.weight_entry.delete(0, "end")
        self.weight_entry.insert(0, values[3])
        self.fhwo_var.set(values[4])
        self.favc_var.set(values[5])
        self.fcvc_var.set(str(round(float(values[6]))))
        self.ncp_entry.delete(0, "end")
        self.ncp_entry.insert(0, values[7])
        self.caec_var.set(values[8])
        self.smoke_var.set(values[9])
        self.ch2o_var.set(str(round(float(values[10]))))
        self.scc_var.set(values[11])
        self.faf_var.set(str(round(float(values[12]))))
        self.tue_var.set(str(round(float(values[13]))))
        self.calc_var.set(values[14])
        self.mtrans_var.set(values[15])
        
        # Calculate BMI
        self.calculate_bmi()
        
        # Switch to personal tab
        self.notebook.select(0)
        
        # Log action
        self.log_message(f"Loaded example with target: {values[16]}")


    def run_batch_predictions(self):
        """Run batch predictions on sample data and visualize results"""
        if model is None:
            messagebox.showerror("Error", "No model available for predictions")
            return

        try:
            # Create sample data (normally would use test data from dataset)
            categories = list(ordinal_mapping.keys())
            # Count of actual predictions by category
            actual_counts = {category: 0 for category in categories}

            # Create sample data for visualization
            num_samples = 50
            sample_data = []

            # Generate random samples within reasonable ranges
            for _ in range(num_samples):
                gender = np.random.choice(["Male", "Female"])
                age = np.random.uniform(18, 70)
                height = np.random.uniform(1.5, 2.0)
                weight = np.random.uniform(45, 120)
                bmi = weight / (height * height)

                # Calculate expected category based on BMI
                expected_category = classify_by_bmi(bmi)
                actual_counts[expected_category] += 1

                # Generate other features
                features = np.random.random(10)  # Placeholder for other features

                sample_data.append({
                    "gender": gender,
                    "age": age,
                    "height": height,
                    "weight": weight,
                    "bmi": bmi,
                    "expected_category": expected_category
                })

            # Run predictions
            model_predictions = {category: 0 for category in categories}

            for sample in sample_data:
                # Prepare features for model
                features_raw = np.array([[
                    gender_map.get(sample["gender"], 0),
                    sample["age"],
                    sample["height"],
                    sample["weight"],
                    np.random.choice([0, 1]),  # family history
                    np.random.choice([0, 1]),  # FAVC
                    np.random.uniform(1, 3),   # FCVC
                    np.random.uniform(1, 4),   # NCP
                    np.random.choice([0, 1, 2, 3]),  # CAEC
                    np.random.uniform(1, 3),   # CH2O
                    np.random.choice([0, 1]),  # SCC
                    np.random.uniform(0, 3),   # FAF
                    np.random.uniform(0, 3),   # TUE
                    sample["bmi"]              # BMI
                ]])

                # Normalize features if we have a scaler
                features = features_raw
                if scaler is not None and hasattr(scaler, 'mean_'):
                    features = scaler.transform(features_raw)

                # Get prediction
                prediction = model.predict(features)[0]
                predicted_category = reverse_mapping.get(prediction, "Unknown")
                model_predictions[predicted_category] = model_predictions.get(predicted_category, 0) + 1

            # Create visualization
            self.create_prediction_visualization(actual_counts, model_predictions)

            # Log action
            self.log_message(f"Ran batch predictions on {num_samples} samples")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to run batch predictions: {str(e)}")
            self.log_message(f"ERROR: Failed to run batch predictions - {str(e)}")


    def create_prediction_visualization(self, actual_counts, predicted_counts):
        """Create a visualization of the prediction results"""
        # Clear existing visualization
        for widget in self.viz_frame.winfo_children():
            widget.destroy()

        # Create figure
        fig, ax = plt.subplots(figsize=(8, 4))
        
        # Categories
        categories = list(ordinal_mapping.keys())
        x = np.arange(len(categories))
        width = 0.35
        
        # Actual vs predicted counts
        actual_values = [actual_counts.get(category, 0) for category in categories]
        predicted_values = [predicted_counts.get(category, 0) for category in categories]
        
        # Create bars
        ax.bar(x - width/2, actual_values, width, label='BMI Classification')
        ax.bar(x + width/2, predicted_values, width, label='Model Prediction')
        
        # Customize chart
        ax.set_xlabel('Obesity Categories')
        ax.set_ylabel('Count')
        ax.set_title('BMI Classification vs. Model Prediction')
        ax.set_xticks(x)
        ax.set_xticklabels([cat.replace('_', ' ') for cat in categories], rotation=45, ha='right')
        ax.legend()
        
        fig.tight_layout()
        
        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.viz_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)


    def log_message(self, message):
        """Add a message to the logs with timestamp"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.logs_text.insert("end", f"[{timestamp}] {message}\n")
        self.logs_text.see("end")


    def save_logs(self):
        """Save logs to a file"""
        file_path = filedialog.asksaveasfilename(
            title="Save Logs",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt")]
        )
        
        if not file_path:
            return

        try:
            with open(file_path, "w") as file:
                file.write(self.logs_text.get(1.0, "end"))
            messagebox.showinfo("Success", "Logs saved successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save logs: {str(e)}")
    
   
if __name__ == "__main__":
    app = ObesityPredictionApp()
    app.mainloop()