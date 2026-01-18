import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from io import BytesIO
import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
import requests  # لتحميل الملفات من GitHub

st.set_page_config(
    page_title="نظام التحليل الذكي للسباحة",
    layout="wide",
    page_icon="🏊‍♂️"
)

st.title("🤖 نظام التحليل والتنبؤ الذكي لأداء السباحين (عام)")

with st.expander("📖 دليل الاستخدام - كيف تفهم نتائجك؟", expanded=True):
    st.markdown("""
    <div dir="rtl">

    ### مرحباً بك في نظام تحليل أداء السباحة 🏊‍♂️
    هذا النظام مصمم لمساعدتك كمدرب أو سباح على فهم مستوى الكفاءة وتوقع الأزمنة المستقبلية بدقة عالية.

    ---

    #### 1️⃣ إدخال البيانات وتحديثها
    * **الجدول الذكي:** يمكنك إضافة بيانات السباحين مباشرة، وسيقوم النظام بتخصيص رقم تعريفي (ID) لكل صف جديد تلقائياً.
    * **الحفظ الفوري:** أي تعديل تقوم به في الجدول يتم حفظه في ذاكرة النظام المؤقتة فوراً دون الحاجة لضغط أزرار حفظ.
    * **المسافات المتاحة:** يدعم النظام المسافات من 50م وحتى 1500م.

    #### 2️⃣ آلية التنبؤ بالأزمنة
    يعتمد النظام على عدة طرق لضمان دقة النتائج:
    * **الذكاء الاصطناعي:** استخدام نماذج مدربة على بيانات سباحة عالمية لتوقع الزمن المثالي حسب العمر والنوع.
    * **التعلّم الشخصي:** إذا أدخلت عدة مسافات لنفس السباح، سيتعرف النظام على نمط تطوره الشخصي ويتنبأ بناءً عليه.
    * **معادلة التعب:** في حال نقص البيانات، يتم استخدام معامل التعب الرياضي (Fatigue Factor) لتوقع الزمن.

    #### 3️⃣ مستويات الشدة التدريبية
    يقوم النظام بتقسيم الزمن المتوقع إلى مستويات تساعدك في وضع خطة التدريب:
    * **مستوى (100% - 95%):** يمثل سرعة المنافسة والسباق.
    * **مستوى (90% - 85%):** يمثل تدريبات التحمل اللاهوائي.
    * **مستوى (80% - 65%):** يمثل التحمل الهوائي والاستشفاء.

    #### 4️⃣ مؤشرات تقييم الأداء
    * **نسبة هبوط السرعة:** توضح مدى قدرة السباح على الحفاظ على سرعته (كلما قلّت النسبة، زادت قوة تحمل السباح).
    * **الكفاءة الفنية:** تقييم شامل لمستوى السباح يجمع بين السرعة القصوى وقدرة التحمل.

    #### 5️⃣ التقارير والمقارنة
    * **تصدير البيانات:** يمكنك تحميل التقرير كاملاً بصيغة إكسيل (Excel) بضغطة زر.
    * **مقارنة الأداء:** عند رفع ملف قديم، سيقارن النظام بين النتائج الحالية والسابقة لتوضيح نسبة التطور باللون الأخضر أو التراجع باللون الأحمر.

    ---
    **💡 نصيحة للمدربين:** استخدم "معامل التعب" في القائمة الجانبية بدقة؛ سباح المسافات الطويلة يحتاج معامل منخفض (1.02 - 1.04)، بينما سباح السرعة يحتاج معامل أعلى (1.08 فأكثر).

    </div>
    """, unsafe_allow_html=True)
st.markdown("---")

today = datetime.now().strftime("%Y-%m-%d")

effort_levels = {
    "100%": 1.00,
    "95%": 0.95,
    "90%": 0.90,
    "85%": 0.85,
    "80%": 0.80,
    "65%": 0.65
}
effort_list = list(effort_levels.keys())

# المسافات المتاحة بناءً على الموديلات
available_distances = [50,100, 200, 400, 800, 1500]

# روابط GitHub الخام للموديلات والفيوتشرز (استبدل بـ repo الخاص بك)
GITHUB_BASE_URL = "https://github.com/BasemAkram/swim-analyzer/tree/main/models"  # استبدل بالرابط الخاص بك
model_urls = {
    100: GITHUB_BASE_URL + "model_100m.pkl",
    200: GITHUB_BASE_URL + "model_200m.pkl",
    400: GITHUB_BASE_URL + "model_400m.pkl",
    800: GITHUB_BASE_URL + "model_800m.pkl",
    1500: GITHUB_BASE_URL + "model_1500m.pkl"
}
features_urls = {
    100: GITHUB_BASE_URL + "features_100m.pkl",
    200: GITHUB_BASE_URL + "features_200m.pkl",
    400: GITHUB_BASE_URL + "features_400m.pkl",
    800: GITHUB_BASE_URL + "features_800m.pkl",
    1500: GITHUB_BASE_URL + "features_1500m.pkl"
}

if "data" not in st.session_state:
    st.session_state.data = pd.DataFrame({
        "ID": ["SW-01", "SW-02"],
        "المسافة الحالية (م)": [100.0, 200.0],
        "الزمن الحالي (ث)": [60.0, 120.0],
        "المسافة المستهدفة (م)": [200.0, 400.0],
    })
    st.info("تم تحميل البيانات الافتراضية المحلية فقط.")

if "results" not in st.session_state:
    st.session_state.results = None

if "models" not in st.session_state:
    st.session_state.models = {}
    # تحميل الموديلات تلقائياً من GitHub
    for dist in available_distances:
        try:
            response = requests.get(model_urls[dist])
            if response.status_code == 200:
                st.session_state.models[dist] = joblib.load(BytesIO(response.content))
                st.success(f"تم تحميل مودل {dist}م من GitHub!")
            else:
                st.session_state.models[dist] = None
        except:
            st.session_state.models[dist] = None

if "features" not in st.session_state:
    st.session_state.features = {}
    # تحميل الفيوتشرز (افترض أنها dataframes أو arrays)

    for dist in available_distances:
        try:
            response = requests.get(features_urls.get(dist))
            if response.status_code == 200:
                st.session_state.features[dist] = joblib.load(BytesIO(response.content))
            else:
                st.session_state.features[dist] = None
        except:
            st.session_state.features[dist] = None

    # fallback تلقائي للـ50م إذا ما تم تحميله
    if st.session_state.features.get(50) is None:
        nearest = next((d for d in [100,200,400,800,1500] if st.session_state.features.get(d) is not None), None)
        if nearest:
            st.session_state.features[50] = st.session_state.features[nearest]
st.sidebar.header("📂 إدارة الملفات")

uploaded_file = st.sidebar.file_uploader("رفع ملف بيانات (Excel أو CSV) - اختياري", type=['xlsx', 'csv'])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            loaded_df = pd.read_csv(uploaded_file)
        else:
            loaded_df = pd.read_excel(uploaded_file)

        required_cols = ["ID", "المسافة الحالية (م)", "الزمن الحالي (ث)", "المسافة المستهدفة (م)"]
        if all(col in loaded_df.columns for col in required_cols):
            # فلتر المسافات لتكون من المتاحة
            loaded_df = loaded_df[loaded_df["المسافة الحالية (م)"].isin(available_distances)]
            loaded_df = loaded_df[loaded_df["المسافة المستهدفة (م)"].isin(available_distances + [0])]
            st.session_state.data = loaded_df[required_cols]
            st.session_state.editor_data = loaded_df[required_cols].copy()
            st.session_state.results = None
            st.sidebar.success("تم تحميل البيانات بنجاح (مع فلتر المسافات المتاحة)!")
        else:
            st.sidebar.error("الملف المرفوع لا يحتوي على الأعمدة المطلوبة.")
    except Exception as e:
        st.sidebar.error(f"حدث خطأ أثناء تحميل الملف: {e}")

uploaded_old = st.sidebar.file_uploader(
    "رفع بيانات متحللة سابقة (Excel أو CSV) - اختياري",
    type=['xlsx','csv'],
    key="old_data"
)

if uploaded_old is not None:
    try:
        if uploaded_old.name.endswith('.csv'):
            old_results_df = pd.read_csv(uploaded_old)
        else:
            old_results_df = pd.read_excel(uploaded_old)
        st.session_state.old_results = old_results_df
        st.sidebar.success("تم تحميل البيانات القديمة!")
    except Exception as e:
        st.sidebar.error(f"حدث خطأ أثناء تحميل الملف القديم: {e}")


st.sidebar.divider()
st.sidebar.header("⚙️ الإعدادات الفنية")
fatigue = st.sidebar.slider("معامل التعب البدني (b) الافتراضي", 1.02, 1.10, 1.06, 0.01)
st.sidebar.info(f"📅 التاريخ: {today}")

st.title("🏊‍♂️ النظام التحليلي العام لتقييم الأداء")

st.subheader("📋 مدخلات البيانات العامة")

# 1. تهيئة البيانات في session_state إذا لم تكن موجودة
if "main_df" not in st.session_state:
    st.session_state.main_df = pd.DataFrame({
        "ID": ["SW-1"],
        "الاسم": ["سباح 1"], # العمود الجديد
        "المسافة الحالية (م)": [100],
        "الزمن الحالي (ث)": [60.0],
        "المسافة المستهدفة (م)": [200]
    })

# 2. دالة معالجة التغييرات (الحل السحري)
def handle_editor_changes():
    changes = st.session_state["swimming_editor"]
    df = st.session_state.main_df.copy()

    # أ. معالجة الصفوف المعدلة (تعمل تلقائياً مع أي عمود جديد)
    for row_idx, updated_values in changes["edited_rows"].items():
        for col, val in updated_values.items():
            df.at[df.index[row_idx], col] = val

    # ب. معالجة الصفوف المضافة (إضافة ID واسم افتراضي)
    for new_row in changes["added_rows"]:
        new_row["ID"] = f"SW-{len(df) + 1}"
        if "الاسم" not in new_row:
            new_row["الاسم"] = f"سباح جديد"

        new_df = pd.DataFrame([new_row])
        df = pd.concat([df, new_df], ignore_index=True)

    # ج. معالجة الصفوف المحذوفة
    indices_to_drop = [df.index[i] for i in changes["deleted_rows"]]
    df = df.drop(indices_to_drop).reset_index(drop=True)

    st.session_state.main_df = df

# 3. عرض المحرر
st.subheader("🏊‍♂️ محرر بيانات السباحين الذكي")

# ملاحظة: نمرر البيانات من session_state مباشرة
st.data_editor(
    st.session_state.main_df,
    key="swimming_editor",
    on_change=handle_editor_changes,
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "ID": st.column_config.TextColumn("ID", disabled=True),
        # تم حذف placeholder لتجنب الخطأ
        "الاسم": st.column_config.TextColumn("اسم السباح"),
        "المسافة الحالية (م)": st.column_config.SelectboxColumn(options=[50, 100, 200, 400, 800, 1500], required=True),
        "الزمن الحالي (ث)": st.column_config.NumberColumn(min_value=0, required=True),
        "المسافة المستهدفة (م)": st.column_config.SelectboxColumn(options=[50, 100, 200, 400, 800, 1500], required=False),
    }
)
# 4. عرض البيانات للتأكد من حفظها
st.divider()
st.dataframe(st.session_state.main_df)


if st.button("🚀 تشغيل التحليل العام", use_container_width=True, type="primary"):
    df = st.session_state.main_df.copy()

    numeric_cols = [
        "المسافة الحالية (م)",
        "الزمن الحالي (ث)",
        "المسافة المستهدفة (م)"
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df = df[(df["المسافة الحالية (م)"] > 0) & (df["الزمن الحالي (ث)"] > 0)]

    df["السرعة الحالية (م/ث)"] = (df["المسافة الحالية (م)"] / df["الزمن الحالي (ث)"]).round(2)

    # -----------------------------
    # دالة التنبؤ بالزمن لكل سباح
    # -----------------------------
    def predict_time(row, fatigue_default):
        d1 = float(row["المسافة الحالية (م)"])
        t1 = float(row["الزمن الحالي (ث)"])
        d2 = float(row["المسافة المستهدفة (م)"]) if row["المسافة المستهدفة (م)"] > 0 else d1

        b = fatigue_default

        # الحالة 1: نفس المسافة
        if d1 == d2:
            t_pred = t1
        else:
            # الحالة 2: مسافة مختلفة → استخدام موديل إن وجد
            model = st.session_state.models.get(int(d2), None)
            if model:
                log_d1 = np.log(d1)
                t_model_current = np.exp(model.predict([[log_d1]])[0])
                log_d2 = np.log(d2)
                t_model_target = np.exp(model.predict([[log_d2]])[0])

                # scale_factor محدود لتجنب نتائج متطرفة
                scale_factor = t1 / t_model_current if t_model_current > 0 else 1
                scale_factor = min(max(scale_factor, 0.7), 1.3)
                t_pred = t_model_target * scale_factor
            else:
                # استخدام المعادلة النسبية
                t_pred = t1 * (d2 / d1) ** b

        # حماية إضافية: لا يقل عن t1 ولا يزيد عن 3 أضعاف
        t_pred = max(t_pred, t1)
        t_pred = min(t_pred, t1 * 3)

        # حساب الأزمنة والسرعات لكل مستوى كفاءة
        times_per_level = {}
        speeds_per_level = {}
        for lvl, coeff in effort_levels.items():
            t_lvl = t_pred / coeff
            s_lvl = d2 / t_lvl if t_lvl > 0 else 0
            times_per_level[lvl] = t_lvl
            speeds_per_level[lvl] = s_lvl

        speed_pred = d2 / t_pred if t_pred > 0 else 0

        return t_pred, d2, b, speed_pred, times_per_level, speeds_per_level

    # -----------------------------
    # تحديث النتائج لكل السباحين
    # -----------------------------
    results = []
    for idx, row in df.iterrows():
        t_100, target_d, used_b, speed_pred, times_per_level, speeds_per_level = predict_time(row, fatigue)
        new_row = row.copy()
        new_row["target_d"] = target_d
        new_row["سرعة متوقعة بالمعادلة (م/ث)"] = round(speed_pred, 2)
        new_row["معامل التعب المستخدم (b)"] = round(used_b, 2)

        for lvl in times_per_level:
            new_row[f"زمن متوقع {lvl} (ث)"] = round(times_per_level[lvl], 2)
            new_row[f"سرعة متوقعة {lvl} (م/ث)"] = round(speeds_per_level[lvl], 2)
            new_row[f"نسبة الزمن {lvl} (%)"] = round(times_per_level[lvl] / t_100 * 100, 1) if t_100 > 0 else 0

        results.append(new_row)

    res_df = pd.DataFrame(results)

    # -----------------------------
    # حساب المؤشرات المشتقة
    # -----------------------------
    res_df["نسبة هبوط السرعة (%)"] = (
            (res_df["السرعة الحالية (م/ث)"] - res_df["سرعة متوقعة بالمعادلة (م/ث)"])
            / res_df["السرعة الحالية (م/ث)"] * 100
    ).round(1).fillna(0)

    res_df["درجة السرعة"] = (
            res_df["السرعة الحالية (م/ث)"] / res_df["السرعة الحالية (م/ث)"].max() * 100
    ).round(1).fillna(0)

    res_df["درجة التحمل"] = (
            100 - res_df["نسبة هبوط السرعة (%)"]
    ).clip(0, 100).round(1)

    # حذف الأعمدة الزائدة
    columns_to_drop = [
        "زمن 100% (ث)",
        "الكفاءة الفنية الواقعية",
        "درجة التحمل",
        "نسبة هبوط السرعة (%)"
    ]
    res_df = res_df.drop(columns=[col for col in columns_to_drop if col in res_df.columns])

    st.session_state.results = res_df


if st.session_state.results is not None:
    res = st.session_state.results

    st.divider()
    st.header("📊 التقرير العام")

    # نسخ البيانات للعمل عليها مؤقتاً
    res_temp = res.copy()

    # -----------------------------
    # حساب الأعمدة المؤقتة للعرض فقط
    # -----------------------------
    res_temp["نسبة هبوط السرعة (%)"] = (
            (res_temp["السرعة الحالية (م/ث)"] - res_temp.get("سرعة متوقعة بالمعادلة (م/ث)", 0))
            / res_temp["السرعة الحالية (م/ث)"] * 100
    ).clip(0, 100).fillna(0)

    res_temp["درجة السرعة"] = (
            res_temp["السرعة الحالية (م/ث)"] / res_temp["السرعة الحالية (م/ث)"].max() * 100
    ).fillna(0)

    res_temp["درجة التحمل"] = 100 - res_temp["نسبة هبوط السرعة (%)"]

    res_temp["كفاءة فنية مؤقتة"] = (
            res_temp["درجة السرعة"] * 0.6 + res_temp["درجة التحمل"] * 0.4
    ).fillna(0)

    # -----------------------------
    # عرض متوسطات عامة
    # -----------------------------
    team_avg_speed = res_temp["السرعة الحالية (م/ث)"].mean()
    team_avg_eff = res_temp["كفاءة فنية مؤقتة"].mean()
    team_avg_drop = res_temp["نسبة هبوط السرعة (%)"].mean()

    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.metric("متوسط السرعة", f"{team_avg_speed:.2f} م/ث")
    with col_m2:
        st.metric("متوسط الكفاءة", f"{team_avg_eff:.1f}%")
    with col_m3:
        st.metric("متوسط هبوط السرعة", f"{team_avg_drop:.1f}%")

    # -----------------------------
    # رسم Box plot لمستويات الكفاءة
    # -----------------------------
    fig_perf = go.Figure()
    for lvl in effort_list:
        if f"زمن متوقع {lvl} (ث)" in res_temp.columns:
            fig_perf.add_trace(go.Box(
                y=res_temp[f"زمن متوقع {lvl} (ث)"],
                name=lvl
            ))

    fig_perf.update_layout(
        title="⏱️ توزيع الأزمنة المتوقعة لمستويات الكفاءة",
        yaxis_title="الزمن (ث)",
        height=400
    )
    st.plotly_chart(fig_perf, use_container_width=True)

    # -----------------------------
    # رسم Scatter لمقارنة السرعة ونسبة الهبوط
    # -----------------------------
    fig_scatter = go.Figure()
    fig_scatter.add_trace(go.Scatter(
        x=res_temp["نسبة هبوط السرعة (%)"],
        y=res_temp["السرعة الحالية (م/ث)"],
        mode='markers',
        marker=dict(color='blue', size=10)
    ))

    fig_scatter.update_layout(
        xaxis_title="نسبة هبوط السرعة (%) (الأقل أفضل)",
        yaxis_title="السرعة الحالية (م/ث) (الأعلى أفضل)",
        height=400
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.divider()
    st.header("📥 مركز التقارير")

    file_name = st.text_input("اسم الملف لحفظ البيانات:", "تقرير_عام")


    def sec_to_min_sec_ms(seconds, decimals=2):
        if pd.isna(seconds):
            return ""
        seconds = float(seconds)
        m = int(seconds // 60)
        s = seconds % 60
        return f"{m}:{s:0{2 + decimals + 1}.{decimals}f}"


    def to_excel(df):
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='SwimmingAnalysis')
        return output.getvalue()


    t_report, t_download = st.tabs(["📋 عرض التقرير", "💾 تحميل"])

    with t_report:
        df = st.session_state.results.copy()

        # تحويل أعمدة الزمن من ثواني إلى دقيقة:ثانية.جزء من الثانية
        time_cols = [col for col in df.columns if "(ث)" in col]

        for col in time_cols:
            df[col.replace("(ث)", "(د:ث.ج)")] = df[col].apply(sec_to_min_sec_ms)

        speed_cols = [f"سرعة متوقعة {e} (م/ث)" for e in effort_list if f"سرعة متوقعة {e} (م/ث)" in df.columns]
        if speed_cols:
            df["السرعة المتوقعة (م/ث)"] = df[speed_cols].max(axis=1)
        else:
            df["السرعة المتوقعة (م/ث)"] = 0

        # ابحث عن هذا الجزء في الكود الخاص بك وقم بتعديله
        report_cols = [
                          "ID",
                          "الاسم",
                          "المسافة الحالية (م)",
                          "الزمن الحالي (د:ث.ج)",
                          "المسافة المستهدفة (م)"
                      ] + [f"زمن متوقع {e} (د:ث.ج)" for e in effort_list] + [
                          "السرعة الحالية (م/ث)",
                          "معامل التعب المستخدم (b)"
                      ]

        # بعد ذلك سيتم تحديث التقرير والتحميل تلقائياً
        full_report = df[report_cols]

        st.dataframe(full_report, use_container_width=True, height=400)

    with t_download:
        c1, c2 = st.columns(2)

        excel_data = to_excel(full_report)

        c1.download_button(
            "📊 تحميل التقرير الكامل (Excel)",
            excel_data,
            f"{file_name}_{today}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

        # تحميل النموذج الحالي (لرفعه إلى GitHub يدوياً إذا أردت)
        if st.session_state.models:
            for dist, model in st.session_state.models.items():
                if model:
                    model_data = BytesIO()
                    joblib.dump(model, model_data)
                    st.download_button(
                        f"💾 تحميل مودل {dist}م (لرفعه إلى GitHub)",
                        model_data.getvalue(),
                        f"model_{dist}m.pkl",
                        use_container_width=True
                    )

    st.divider()
    st.header("⚡ مقارنة القيم القديمة والجديدة")

    if "old_results" in st.session_state and st.session_state.old_results is not None:
        if st.session_state.results is not None:
            old_data = st.session_state.old_results.copy()
            new_data = st.session_state.results.copy()

            compare_cols = [
                "الزمن الحالي (ث)",
                "زمن 100% (ث)",
                "السرعة الحالية (م/ث)",
                "السرعة المتوقعة (م/ث)",
                "الكفاءة الفنية الواقعية",
                "نسبة هبوط السرعة (%)",
                "درجة التحمل"
            ]

            compare_cols = [col for col in compare_cols if col in old_data.columns and col in new_data.columns]

            compare_df = pd.DataFrame()
            compare_df["ID"] = new_data["ID"]

            for col in compare_cols:
                compare_df[f"{col} (قديم)"] = old_data[col]
                compare_df[f"{col} (جديد)"] = new_data[col]


            def highlight_columns(x):
                color_map = {}
                for col in compare_df.columns:
                    if "(قديم)" in col:
                        color_map[col] = 'background-color: lightblue'
                    elif "(جديد)" in col:
                        color_map[col] = 'background-color: lightgreen'
                return pd.DataFrame([color_map] * len(compare_df), index=compare_df.index)


            st.dataframe(compare_df.style.apply(highlight_columns, axis=None), use_container_width=True, height=600)
        else:
            st.info("📌 لم يتم تشغيل التحليل بعد.")
    else:
        st.info("📌 لم يتم رفع بيانات سابقة.")