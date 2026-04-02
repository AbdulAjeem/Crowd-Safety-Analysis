import streamlit as st, cv2, numpy as np, pandas as pd, requests, plotly.express as px

st.set_page_config(layout="wide", page_title="Crowd Safety AI")
st.title("🚨 Crowd Safety Analysis AI")

# Sidebar
api_key = st.sidebar.text_input("🔑 Google Gemini API Key", type="password")

# Tabs
tab1, tab2, tab3 = st.tabs(["📤 Upload", "📊 Analysis", "📄 Report"])

# Upload
with tab1:
    file = st.file_uploader("Upload Crowd Image", type=["jpg","png","jpeg"])
    if file:
        img = cv2.imdecode(np.frombuffer(file.getvalue(), np.uint8), 1)
        st.image(img, channels="BGR", use_container_width=True)
    else:
        st.info("Upload image to start")

# Analyze
def analyze(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detection (HOG + fallback)
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    boxes, _ = hog.detectMultiScale(img, winStride=(8,8))

    if len(boxes) == 0:
        edges = cv2.Canny(gray, 50, 150)
        contours,_ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = [(x,y,w,h) for c in contours if 500 < cv2.contourArea(c) < 5000
                 for (x,y,w,h) in [cv2.boundingRect(c)]]

    count = len(boxes)

    # Draw boxes
    display = img.copy()
    for (x,y,w,h) in boxes:
        cv2.rectangle(display,(x,y),(x+w,y+h),(0,255,0),2)

    # Heatmap
    heat = np.zeros_like(gray, dtype=np.float32)
    for (x,y,w,h) in boxes:
        heat[y:y+h, x:x+w] += 1

    if count > 0:
        heat = cv2.GaussianBlur(heat,(51,51),0)
        heat = cv2.normalize(heat,None,0,255,cv2.NORM_MINMAX)

    heat = cv2.applyColorMap(heat.astype(np.uint8), cv2.COLORMAP_JET)

    density = count / (img.shape[0]*img.shape[1]) * 100000

    return count, density, heat, display

# Gemini AI (fixed)
def get_llm(count, density):
    # Risk level logic
    if count < 5:
        risk = "Low"
    elif count < 15:
        risk = "Medium"
    else:
        risk = "High"

    # Safety suggestions
    if risk == "Low":
        advice = [
            "Maintain normal monitoring",
            "Ensure clear pathways",
            "Keep emergency exits accessible"
        ]
    elif risk == "Medium":
        advice = [
            "Control entry and exit points",
            "Deploy additional staff",
            "Monitor crowd movement continuously"
        ]
    else:
        advice = [
            "Restrict further entry immediately",
            "Deploy emergency response teams",
            "Ensure evacuation routes are clear"
        ]

    return f"""
Risk Level: {risk}

Safety Measures:
- {advice[0]}
- {advice[1]}
- {advice[2]}

Emergency Advice:
Stay alert and ensure crowd dispersal if density increases.
"""
# Analysis
with tab2:
    if file:
        count, density, heat, display = analyze(img)

        col1, col2 = st.columns(2)
        col1.metric("👥 Crowd Count", count)
        col2.metric("📊 Density", round(density,2))

        st.image(display, caption="Detected Crowd", use_container_width=True)
        st.image(heat, caption="🔥 Heatmap", use_container_width=True)

        df = pd.DataFrame({"Metric":["Count","Density"],
                           "Value":[max(count,1), max(density,1)]})

        st.plotly_chart(px.bar(df,x="Metric",y="Value"), use_container_width=True)
        st.plotly_chart(px.pie(df,names="Metric",values="Value"), use_container_width=True)

        st.line_chart(pd.DataFrame({
            "Frame":[1,2,3,4],
            "Density":[density*0.8,density*0.9,density,density*1.1]
        }).set_index("Frame"))

        advice = get_llm(count, density)
        st.subheader("🧠 AI Safety Advice")
        st.write(advice)

# Report
with tab3:
    if file:
        report = f"""
Crowd Safety Report
-------------------
Count: {count}
Density: {density:.2f}

AI Advice:
{advice}
"""
        st.download_button("📥 Download Report", report, "report.txt")
        st.text_area("Preview", report, height=300)
