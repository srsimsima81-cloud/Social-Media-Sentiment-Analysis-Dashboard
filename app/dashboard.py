import streamlit as st
import pandas as pd
import os
import sys
import base64
import plotly.graph_objects as go

# -------------------------------------------------
# PATH FIX
# -------------------------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

from src.preprocess import clean_text
from src.bert_model import predict_sentiment

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Reputation Radar",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -------------------------------------------------
# DATA LOAD (SAFE PATHS)
# -------------------------------------------------
DATA_PATH = os.path.join(ROOT_DIR, "data", "social_media.csv")
df = pd.read_csv(DATA_PATH)

# -------------------------------------------------
# LIKE IMAGE
# -------------------------------------------------
like_path = os.path.join(ROOT_DIR, "images", "like.png")
with open(like_path, "rb") as f:
    like_base64 = base64.b64encode(f.read()).decode()

# -------------------------------------------------
# CSS 
# -------------------------------------------------
st.markdown("""
<style>

html, body, [class*="css"]{
background: radial-gradient(circle at top left,#180428 0%,#090613 45%,#020204 100%);
color:white;
font-family:Segoe UI;
}

header{visibility:hidden;}
footer{visibility:hidden;}

/* ---------------- HERO ---------------- */
.hero{text-align:center;margin-top:10px;}

.radar{
font-size:42px;
color:#00eaff;
text-shadow:0 0 10px #00eaff,0 0 20px #00eaff;
animation:pulse 1.5s infinite;
}

.title{
font-size:78px;
font-weight:900;
letter-spacing:4px;
position:relative;
color:transparent;
-webkit-text-stroke:2px white;
animation:glitch .35s 1.2s forwards, filltitle 1.2s 1.55s forwards;
}

.title::after{
content:"❤";
position:absolute;
left:50%;
top:50%;
transform:translate(-50%,-50%) scale(0);
font-size:60px;
color:#ff2d95;
text-shadow:0 0 12px #ff2d95,0 0 24px #ff2d95;
animation:heartflash .8s .6s forwards;
}

.subtitle{font-size:22px;margin-top:-10px;color:#d7d7d7;}

.line{
width:70%;
height:2px;
margin:18px auto;
background:linear-gradient(90deg,transparent,#00eaff,#ff2d95,#00eaff,transparent);
box-shadow:0 0 10px #00eaff,0 0 18px #ff2d95;
}

/* ---------------- FLOATING ICONS (FIXED) ---------------- */
/* ---------------- FLOAT ICONS ---------------- */

.float{
position:fixed;
bottom:-80px;
z-index:999;
animation:riseRandom 5s linear forwards;
opacity:0;
pointer-events:none;
filter: blur(0px);
}

.float-img{
width:52px;   
height:52px;
object-fit:contain;

filter:
drop-shadow(0 0 6px #1877f2)
drop-shadow(0 0 14px #1877f2)
drop-shadow(0 0 22px #1877f2);
transform: rotate(0deg);
}

.love{
width:48px;   
height:48px;
border-radius:50%;
display:flex;
align-items:center;
justify-content:center;
font-size:20px;

background: radial-gradient(circle at 30% 30%,#ff7bc5,#ff2d95);

box-shadow:
0 0 10px #ff2d95,
0 0 22px #ff2d95,
0 0 35px #ff2d95;
}

/* random positions */
.f1{left:6%; animation-delay:0.2s;}
.f2{left:18%; animation-delay:1.1s;}
.f3{left:29%; animation-delay:0.6s;}
.f4{left:44%; animation-delay:2.0s;}
.f5{left:57%; animation-delay:1.4s;}
.f6{left:71%; animation-delay:2.6s;}
.f7{left:83%; animation-delay:3.0s;}
.f8{left:91%; animation-delay:1.9s;}

/* TRUE RANDOM MOTION FEEL */
@keyframes riseRandom{

0%{
bottom:-80px;
opacity:0;
transform:translateX(0) scale(.6) rotate(0deg);
}

10%{opacity:1;}

25%{transform:translateX(25px) scale(1) rotate(10deg);}
45%{transform:translateX(-18px) scale(1.08) rotate(-8deg);}
65%{transform:translateX(20px) scale(.95) rotate(12deg);}
85%{transform:translateX(-22px) scale(.9) rotate(-10deg);}

100%{
bottom:110%;
opacity:0;
transform:translateX(10px) scale(.8) rotate(15deg);
}
}

/* ---------------- CARDS ---------------- */
.card{
background:rgba(255,255,255,.03);
border:1px solid rgba(0,255,255,.22);
border-radius:18px;
padding:22px;
text-align:center;
backdrop-filter:blur(12px);
box-shadow:0 0 14px rgba(0,255,255,.15),0 0 24px rgba(255,45,149,.10);
}

/* ---------------- INPUT ---------------- */
.stTextArea textarea{
background:#0f172a !important;
color:white !important;
border:1px solid #00eaff !important;
}

/* ---------------- BUTTON ---------------- */
.stButton>button{
width:100%;
background:linear-gradient(90deg,#00eaff,#ff2d95);
color:white;
font-weight:900;
border-radius:14px;
}

/* ---------------- RESULT ---------------- */
.result{
text-align:center;
font-size:28px;
font-weight:900;
padding:22px;
border-radius:18px;
margin-top:12px;
color:white;

border:1px solid rgba(255,255,255,0.18);
backdrop-filter: blur(12px);


box-shadow:
0 0 10px rgba(0,255,179,0.25),
0 0 25px rgba(0,212,255,0.15),
0 0 40px rgba(255,23,68,0.10);


animation: neonPulse 2.5s infinite ease-in-out;
}
            
@keyframes neonPulse {
0% {
    box-shadow:
    0 0 8px rgba(0,255,179,0.2),
    0 0 18px rgba(0,212,255,0.1),
    0 0 30px rgba(255,23,68,0.08);
}

50% {
    box-shadow:
    0 0 18px rgba(0,255,179,0.4),
    0 0 35px rgba(0,212,255,0.25),
    0 0 55px rgba(255,23,68,0.15);
}

100% {
    box-shadow:
    0 0 8px rgba(0,255,179,0.2),
    0 0 18px rgba(0,212,255,0.1),
    0 0 30px rgba(255,23,68,0.08);
}
}

/* ---------------- ANIMATIONS ---------------- */
@keyframes pulse{50%{transform:scale(1.1);}}

@keyframes glitch{
0%{text-shadow:2px 0 #00eaff,-2px 0 #ff2d95;}
50%{text-shadow:-3px 0 #00eaff,3px 0 #ff2d95;}
}

@keyframes filltitle{
to{
color:white;
-webkit-text-stroke:1px transparent;
text-shadow:0 0 18px #00eaff,0 0 28px #ff2d95;
}
}

@keyframes heartflash{
0%{transform:translate(-50%,-50%) scale(0);}
50%{transform:translate(-50%,-50%) scale(1.6);}
100%{transform:translate(-50%,-50%) scale(0);}
}

</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# FLOATING ICONS 
# -------------------------------------------------
st.markdown(f"""
<div class="float f1"><img src="data:image/png;base64,{like_base64}" class="float-img"></div>
<div class="float f2"><div class="love">❤</div></div>
<div class="float f3"><img src="data:image/png;base64,{like_base64}" class="float-img"></div>
<div class="float f4"><div class="love">❤</div></div>
<div class="float f5"><img src="data:image/png;base64,{like_base64}" class="float-img"></div>
<div class="float f6"><div class="love">❤</div></div>
<div class="float f7"><img src="data:image/png;base64,{like_base64}" class="float-img"></div>
<div class="float f8"><div class="love">❤</div></div>
""", unsafe_allow_html=True)

# -------------------------------------------------
# HERO
# -------------------------------------------------
st.markdown("""
<div class="hero">
<div class="radar">🌐</div>
<div class="title">REPUTATION RADAR</div>
<div class="subtitle">Track Public Emotion at Scale</div>
<div class="line"></div>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------
# METRICS
# -------------------------------------------------
positive = len(df[df["sentiment"]=="positive"])
negative = len(df[df["sentiment"]=="negative"])
neutral  = len(df[df["sentiment"]=="neutral"])
score = round((positive / len(df)) * 100, 2)

c1,c2,c3,c4 = st.columns(4)
c1.markdown(f"<div class='card'><h4>Total</h4><h1>{len(df)}</h1></div>", unsafe_allow_html=True)
c2.markdown(f"<div class='card'><h4>Positive</h4><h1>{positive}</h1></div>", unsafe_allow_html=True)
c3.markdown(f"<div class='card'><h4>Negative</h4><h1>{negative}</h1></div>", unsafe_allow_html=True)
c4.markdown(f"<div class='card'><h4>Score</h4><h1>{score}%</h1></div>", unsafe_allow_html=True)

# -------------------------------------------------
# INPUT + BERT
# -------------------------------------------------
st.subheader("📝 Analyze Comment")

text = st.text_area("Enter text")
btn = st.button("Analyze")

if btn and text.strip():

    cleaned = clean_text(text)
    label, conf = predict_sentiment(cleaned)

    color = {
        "positive": "#00ff0490",
        "negative": "#ff2d2dcd",
        "neutral": "#0080ff"
    }.get(label, "#00c8ff")

    emoji = {
        "positive": "✨",
        "negative": "⚠",
        "neutral": "🔘"
    }.get(label, "🔘")

    st.markdown(f"""
    <div class='result' style='background:{color};'>
    {emoji} {label.upper()}<br>
    Confidence: {round(conf*100,2)}%
    </div>
    """, unsafe_allow_html=True)
# -------------------------------------------------
# COMMAND CENTER TITLE
# -------------------------------------------------
st.markdown("""
<div style="font-size:30px;font-weight:900;margin:20px 0;
text-shadow:0 0 10px #00eaff,0 0 20px #ff2d95;">
⚡ SENTIMENT COMMAND CENTER
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------
# DATA
# -------------------------------------------------
count_df = df["sentiment"].value_counts().reset_index()
count_df.columns = ["sentiment","count"]

labels = count_df["sentiment"].tolist()
vals = count_df["count"].tolist()

colors = {"positive":"#0f6b47","negative":"#7a1622","neutral":"#114e73"}
glow = {"positive":"#00ff99","negative":"#ff3355","neutral":"#00d9ff"}

col1,col2 = st.columns(2)


with col1:

    fig1 = go.Figure()

    # glow ring
    fig1.add_trace(go.Pie(
        labels=labels,
        values=vals,
        hole=0.6,
        marker=dict(colors=[glow[x] for x in labels]),
        opacity=0.25,
        showlegend=False
    ))

    # main ring (WHITE NEON OUTLINE FIX)
    fig1.add_trace(go.Pie(
        labels=labels,
        values=vals,
        hole=0.75,
        marker=dict(
            colors=[colors[x] for x in labels],
            line=dict(color="white", width=3)   
        ),
        textinfo="label+percent",
        textfont=dict(color="white"),
        showlegend=False
    ))

    st.plotly_chart(fig1, use_container_width=True)

# ---------------- BAR (NEON DARK FIXED) ----------------
with col2:

    fig2 = go.Figure()

    for i in labels:

        val = vals[labels.index(i)]

        
        fig2.add_trace(go.Bar(
            x=[i],
            y=[val],
            marker=dict(
                color=glow[i],
                opacity=0.25,
                line=dict(color=glow[i], width=12)  # BIG glow ring
            ),
            hoverinfo="skip",
            showlegend=False
        ))

        
        fig2.add_trace(go.Bar(
            x=[i],
            y=[val],
            marker=dict(
                color=colors[i],
                line=dict(color="white", width=2)
            ),
            text=[val],
            textposition="outside",
            textfont=dict(color="white", size=16),
            showlegend=False
        ))

    fig2.update_layout(

        barmode="overlay",   

        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",

        font_color="white",

        xaxis=dict(
            showgrid=False,
            linecolor="#00eaff",
            tickfont=dict(color="#00eaff", size=14)
        ),

        yaxis=dict(
            gridcolor="rgba(0,255,255,.15)",
            linecolor="#00eaff",
            tickfont=dict(color="#00eaff", size=14)
        )
    )

    st.plotly_chart(fig2, use_container_width=True)

# -------------------------------------------------
st.dataframe(df.head(10))
st.success("Reputation Radar Live ✔")