import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from io import BytesIO
from datetime import datetime

# --- 1. 기본 설정 및 상수 정의 ---
st.set_page_config(
    page_title="손익분석_기조실",
    page_icon="📊",
    layout="wide"
)

# CSS 외부 파일에서 읽어오도록 변경됨 (style.css)
def inject_custom_css():
    """
    외부 style.css 파일을 읽어서 Streamlit 앱에 적용하는 함수
    style.css 파일이 파이썬 파일과 같은 디렉토리에 있어야 합니다.
    """
    try:
        # style.css 파일 경로 (파이썬 파일과 같은 디렉토리)
        import os
        css_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'style.css')
        
        # CSS 파일 읽기
        with open(css_file_path, 'r', encoding='utf-8') as f:
            custom_css = f.read()
        
        # CSS 적용
        st.markdown(f"<style>{custom_css}</style>", unsafe_allow_html=True)
        
    except FileNotFoundError:
        # style.css 파일이 없을 경우 경고 표시
        st.sidebar.warning("⚠️ style.css 파일을 찾을 수 없습니다. 기본 스타일로 표시됩니다.")
    except Exception as e:
        st.sidebar.error(f"CSS 로드 중 오류 발생: {e}")

# ----------------------------------------------------------------


# 비용 컬럼 목록 (generate_sample_data.py 파일 기준)
COST_COLUMNS = [
    '관리자A', '관리자B', '관리자C', '강사A', '강사B', '강사C', '강사D',
    '4대보험근로자', '4대보험강사D', '퇴직추계', '해지미정산', '인센티브직접', '인센티브간저',
    '경비', '본사급여', '본사4대보험', '본사퇴직추계', '셔틀', '동승자',
    '임차A', '임차B', '임차C', '임차D', '관리비A', '관리비B', '관리비C', '관리비D',
    '청소용역A', '청소용역B', '청소용역C', '청소용역D', '복구충당',
    '공통감가비A', '캠퍼스감가비B', '관별감가비B', '공통감가비B',
    '기타1', '제경비', '카드매출수수료', '공기청정기', '정수기', '캡스', '복합기', 'LMS',
    '관마케팅', '캠퍼스마케팅', '관기타2', '캠퍼스기타2'
]

# 비용 카테고리 그룹핑 (AI 분석용)
COST_CATEGORIES = {
    '인건비': ['관리자A', '관리자B', '관리자C', '강사A', '강사B', '강사C', '강사D', '본사급여'],
    '4대보험/퇴직': ['4대보험근로자', '4대보험강사D', '퇴직추계', '본사4대보험', '본사퇴직추계'],
    '인센티브': ['인센티브직접', '인센티브간저'],
    '임차/관리비': ['임차A', '임차B', '임차C', '임차D', '관리비A', '관리비B', '관리비C', '관리비D'],
    '용역/청소': ['청소용역A', '청소용역B', '청소용역C', '청소용역D'],
    '감가상각': ['공통감가비A', '캠퍼스감가비B', '관별감가비B', '공통감가비B', '복구충당'],
    '운영비': ['경비', '셔틀', '동승자', '공기청정기', '정수기', '캡스', '복합기', 'LMS'],
    '마케팅': ['관마케팅', '캠퍼스마케팅'],
    '기타': ['해지미정산', '기타1', '제경비', '카드매출수수료', '관기타2', '캠퍼스기타2']
}

# 통화 형식 지정 함수 (선택된 단위로 나누고 포맷팅)
def format_currency(value, unit_str=" 원", divisor=1):
    if pd.isna(value) or value is None:
        return f"0{unit_str}"
    
    display_value = value / divisor
    
    if divisor == 1:
        # '원' 단위는 정수로 표시
        return f"{int(value):,d}{unit_str}"
    else:
        # '천 원' 이상 단위는 소수점 첫째 자리까지 표시
        return f"{display_value:,.1f}{unit_str}"
        
# 분기 계산 함수
def get_quarter(month_str):
    month = int(month_str)
    if 1 <= month <= 3: return 'Q1'
    if 4 <= month <= 6: return 'Q2'
    if 7 <= month <= 9: return 'Q3'
    if 10 <= month <= 12: return 'Q4'
    return 'N/A'

# --- 엑셀 다운로드 함수 ---
def create_excel_report(df_summary, df_trend, df_cost_analysis=None):
    """
    분석 결과를 엑셀 파일로 생성
    """
    output = BytesIO()
    
    try:
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # 시트 1: 상세 손익 내역
            if df_summary is not None and not df_summary.empty:
                df_summary.to_excel(writer, sheet_name='상세손익내역', index=False)
            
            # 시트 2: 기간별 추이
            if df_trend is not None and not df_trend.empty:
                df_trend.to_excel(writer, sheet_name='기간별추이', index=False)
            
            # 시트 3: 비용 분석 (있을 경우)
            if df_cost_analysis is not None and not df_cost_analysis.empty:
                df_cost_analysis.to_excel(writer, sheet_name='비용항목분석', index=False)
    except Exception as e:
        st.error(f"엑셀 파일 생성 중 오류: {e}")
        return None
    
    output.seek(0)
    return output

# --- HTML 리포트 생성 함수 ---
def create_html_report(
    total_revenue_target, total_cost_target, operating_profit_target,
    total_revenue_comp, total_cost_comp, operating_profit_comp,
    delta_revenue, delta_cost, delta_profit,
    target_label, comparison_year, display_unit, display_divisor,
    df_summary, df_trend, insights
):
    """
    이메일 첨부용 HTML 리포트 생성
    """
    
    # 증감률 계산
    revenue_rate = (delta_revenue / total_revenue_comp * 100) if total_revenue_comp != 0 else 0
    cost_rate = (delta_cost / total_cost_comp * 100) if total_cost_comp != 0 else 0
    profit_rate = (delta_profit / operating_profit_comp * 100) if operating_profit_comp != 0 else 0
    
    # 색상 결정
    profit_color = "#2563eb" if delta_profit >= 0 else "#ef4444"
    revenue_color = "#2563eb" if delta_revenue >= 0 else "#ef4444"
    cost_color = "#ef4444" if delta_cost >= 0 else "#2563eb"
    
    profit_symbol = "+" if delta_profit >= 0 else "△"
    revenue_symbol = "+" if delta_revenue >= 0 else "△"
    cost_symbol = "+" if delta_cost >= 0 else "△"
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>손익 분석 리포트</title>
        <style>
            body {{
                font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f8fafc;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                padding: 40px;
                border-radius: 12px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }}
            h1 {{
                color: #1e40af;
                border-bottom: 3px solid #3b82f6;
                padding-bottom: 10px;
                margin-bottom: 30px;
            }}
            h2 {{
                color: #334155;
                margin-top: 30px;
                margin-bottom: 15px;
            }}
            .metrics {{
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                gap: 20px;
                margin-bottom: 40px;
            }}
            .metric-card {{
                background: white;
                border: 1px solid #e2e8f0;
                border-left: 4px solid #3b82f6;
                border-radius: 12px;
                padding: 20px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }}
            .metric-label {{
                font-size: 0.875rem;
                color: #64748b;
                text-transform: uppercase;
                letter-spacing: 0.05em;
                margin-bottom: 8px;
            }}
            .metric-value {{
                font-size: 2rem;
                font-weight: 700;
                color: #0f172a;
                margin-bottom: 8px;
            }}
            .metric-delta {{
                font-size: 0.95rem;
                font-weight: 600;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 20px;
                font-size: 0.9rem;
            }}
            th {{
                background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
                color: white;
                padding: 12px;
                text-align: left;
                font-weight: 600;
            }}
            td {{
                padding: 10px 12px;
                border-bottom: 1px solid #e2e8f0;
            }}
            tr:nth-child(even) {{
                background-color: #f8fafc;
            }}
            .insight-box {{
                background: #f0f9ff;
                border-left: 4px solid #3b82f6;
                padding: 15px;
                margin: 10px 0;
                border-radius: 8px;
            }}
            .insight-positive {{ border-left-color: #10b981; background: #ecfdf5; }}
            .insight-negative {{ border-left-color: #ef4444; background: #fef2f2; }}
            .footer {{
                margin-top: 40px;
                padding-top: 20px;
                border-top: 2px solid #e2e8f0;
                text-align: center;
                color: #64748b;
                font-size: 0.875rem;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📊 손익 분석 리포트</h1>
            <p><strong>기준 기간:</strong> {target_label}</p>
            <p><strong>비교 기간:</strong> {comparison_year if comparison_year != '선택 안함' else '없음'}</p>
            <p><strong>생성일시:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>🔑 핵심 손익 지표</h2>
            <div class="metrics">
                <div class="metric-card">
                    <div class="metric-label">영업 이익</div>
                    <div class="metric-value">{format_currency(operating_profit_target, display_unit, display_divisor)}</div>
                    <div class="metric-delta" style="color: {profit_color};">
                        {profit_symbol}{abs(delta_profit / display_divisor):.1f}{display_unit} ({profit_symbol}{abs(profit_rate):.1f}%)
                    </div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">총 매출액</div>
                    <div class="metric-value">{format_currency(total_revenue_target, display_unit, display_divisor)}</div>
                    <div class="metric-delta" style="color: {revenue_color};">
                        {revenue_symbol}{abs(delta_revenue / display_divisor):.1f}{display_unit} ({revenue_symbol}{abs(revenue_rate):.1f}%)
                    </div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">총 비용</div>
                    <div class="metric-value">{format_currency(total_cost_target, display_unit, display_divisor)}</div>
                    <div class="metric-delta" style="color: {cost_color};">
                        {cost_symbol}{abs(delta_cost / display_divisor):.1f}{display_unit} ({cost_symbol}{abs(cost_rate):.1f}%)
                    </div>
                </div>
            </div>
    """
    
    # AI 인사이트 추가
    if insights:
        html_content += "<h2>🤖 AI 인사이트</h2>"
        for insight in insights:
            css_class = f"insight-{insight['type']}"
            icon = {'positive': '✅', 'negative': '⚠️', 'neutral': 'ℹ️'}[insight['type']]
            html_content += f"""
            <div class="insight-box {css_class}">
                <strong>{icon} {insight['title']}</strong><br>
                {insight['content']}
            </div>
            """
    
    # 상세 손익 내역 테이블 추가
    if df_summary is not None and not df_summary.empty:
        html_content += "<h2>📊 상세 손익 내역</h2>"
        html_content += df_summary.to_html(index=False, escape=False, classes='data-table')
    
    # 기간별 추이 테이블 추가
    if df_trend is not None and not df_trend.empty:
        html_content += "<h2>📋 기간별 손익 추이</h2>"
        html_content += df_trend.to_html(index=False, escape=False, classes='data-table')
    
    html_content += """
            <div class="footer">
                <p>본 리포트는 손익분석 대시보드에서 자동 생성되었습니다.</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    return html_content

# --- 2. 데이터 로드 및 전처리 캐시 함수 ---
@st.cache_data(show_spinner="엑셀 데이터를 로드하고 분석을 위해 전처리 중입니다...")
def load_data(file):
    try:
        # 파일 읽기 및 기본 전처리
        df_loaded = pd.read_excel(file)
        
        # '년월' 컬럼 전처리
        if '년월' not in df_loaded.columns:
             st.error("필수 컬럼인 '년월'이 누락되었습니다.")
             return None
             
        df_loaded['년월'] = df_loaded['년월'].astype(str)
        df_loaded['년'] = df_loaded['년월'].str[:4]
        df_loaded['월'] = df_loaded['년월'].str[4:6]
        df_loaded['분기'] = df_loaded['월'].apply(get_quarter)
        df_loaded['년분기'] = df_loaded['년'] + ' ' + df_loaded['분기']
        df_loaded['sort_key'] = df_loaded['년월'].astype(int) 
        
        # 필터 컬럼 전처리
        FILTER_COLUMNS = ['수익코드', '캠퍼스', '브랜드', '사업부']
        for col in FILTER_COLUMNS:
            if col not in df_loaded.columns:
                st.error(f"필수 필터 컬럼 '{col}'이 누락되었습니다.")
                return None
            df_loaded[col] = df_loaded[col].fillna('N/A').astype(str)

        # 재무 컬럼 계산
        if '매출액' not in df_loaded.columns:
             st.error("필수 재무 컬럼인 '매출액'이 누락되었습니다.")
             return None
             
        df_loaded['매출액'] = pd.to_numeric(df_loaded['매출액'], errors='coerce').fillna(0)
            
        # 비용 컬럼 존재 확인 및 계산
        all_costs_present = all(col in df_loaded.columns for col in COST_COLUMNS)
        if all_costs_present:
            for col in COST_COLUMNS:
                df_loaded[col] = pd.to_numeric(df_loaded[col], errors='coerce').fillna(0)
                
            df_loaded['총비용'] = df_loaded[COST_COLUMNS].sum(axis=1)
            df_loaded['영업이익'] = df_loaded['매출액'] - df_loaded['총비용']
        else:
            # 비용 컬럼이 없으면 영업이익 = 매출액으로 임시 계산
            df_loaded['총비용'] = 0
            df_loaded['영업이익'] = df_loaded['매출액']
            st.warning("일부 비용 컬럼이 누락되어 '총비용' 및 '영업이익' 계산이 부정확할 수 있습니다.")
            
        return df_loaded
    
    except Exception as e:
        st.error(f"파일을 읽거나 처리하는 도중 오류가 발생했습니다: {e}")
        return None

# 데이터 집계 함수 (display_label 추가)
def aggregate_profit_trend(df_input, time_col, sort_col, is_cumulative, period_label):
    if df_input.empty:
        return None
    
    if time_col == '년분기':
        df_input['time_label'] = df_input['년'] + ' ' + df_input['분기']
        df_agg = df_input.groupby('time_label').agg(
            {'영업이익': 'sum', '매출액': 'sum', sort_col: 'min'}
        ).reset_index().rename(columns={'time_label': time_col})
        df_agg = df_agg.sort_values(sort_col)
        df_agg['display_label'] = df_agg[time_col] # Ex: 2024 Q1
    else: # 월별
        df_agg = df_input.groupby([time_col, sort_col])[['영업이익', '매출액']].sum().reset_index().sort_values(sort_col)
        # '월' 부분을 추출하고, 앞의 0을 제거하여 표시합니다.
        df_agg['display_label'] = df_agg[time_col].str[4:6].str.lstrip('0')

    if is_cumulative:
        df_agg['누적 영업이익'] = df_agg['영업이익'].cumsum()
        df_agg['누적 매출액'] = df_agg['매출액'].cumsum()
        df_agg.drop(columns=['영업이익', '매출액'], inplace=True)
        df_agg.rename(columns={'누적 영업이익': '영업이익', '누적 매출액': '매출액'}, inplace=True)
    
    df_agg['기간'] = period_label
    # time_col과 sort_col은 드롭하고 display_label은 유지
    df_agg.drop(columns=[sort_col, time_col], inplace=True, errors='ignore') 
    return df_agg


# --- NEW: 비용 항목별 분석 함수 ---
def analyze_cost_breakdown(df_target, df_comparison, cost_columns, display_divisor, display_unit):
    """비용 항목별 증감 분석"""
    
    # 주요기간 비용 합계
    target_costs = {}
    for col in cost_columns:
        if col in df_target.columns:
            target_costs[col] = df_target[col].sum()
        else:
            target_costs[col] = 0
    
    # 비교기간 비용 합계
    comp_costs = {}
    if df_comparison is not None and not df_comparison.empty:
        for col in cost_columns:
            if col in df_comparison.columns:
                comp_costs[col] = df_comparison[col].sum()
            else:
                comp_costs[col] = 0
    else:
        for col in cost_columns:
            comp_costs[col] = 0
    
    # 증감 계산
    result = []
    for col in cost_columns:
        target_val = target_costs.get(col, 0)
        comp_val = comp_costs.get(col, 0)
        diff = target_val - comp_val
        
        if comp_val != 0:
            diff_rate = (diff / comp_val) * 100
        else:
            diff_rate = 0 if target_val == 0 else np.inf
            
        result.append({
            '비용항목': col,
            '주요기간': target_val,
            '비교기간': comp_val,
            '증감액': diff,
            '증감률': diff_rate
        })
    
    df_result = pd.DataFrame(result)
    return df_result


# --- NEW: AI 분석 함수 ---
def generate_ai_insights(df_target, df_comparison, cost_columns, cost_categories, 
                         total_revenue_target, total_revenue_comp,
                         operating_profit_target, operating_profit_comp,
                         display_divisor, display_unit):
    """AI 기반 인사이트 생성 (규칙 기반)"""
    
    insights = []
    
    # 1. 전체 실적 요약
    if df_comparison is not None and not df_comparison.empty:
        revenue_diff = total_revenue_target - total_revenue_comp
        profit_diff = operating_profit_target - operating_profit_comp
        
        revenue_rate = (revenue_diff / total_revenue_comp * 100) if total_revenue_comp != 0 else 0
        profit_rate = (profit_diff / operating_profit_comp * 100) if operating_profit_comp != 0 else 0
        
        # 매출 분석
        if revenue_rate > 5:
            insights.append({
                'type': 'positive',
                'title': '📈 매출 성장',
                'content': f"매출액이 전기 대비 {revenue_rate:.1f}% 증가했습니다. ({format_currency(revenue_diff, display_unit, display_divisor)} 증가)"
            })
        elif revenue_rate < -5:
            insights.append({
                'type': 'negative',
                'title': '📉 매출 감소',
                'content': f"매출액이 전기 대비 {abs(revenue_rate):.1f}% 감소했습니다. ({format_currency(abs(revenue_diff), display_unit, display_divisor)} 감소)"
            })
        else:
            insights.append({
                'type': 'neutral',
                'title': '➡️ 매출 유지',
                'content': f"매출액이 전기와 유사한 수준입니다. (변동률: {revenue_rate:.1f}%)"
            })
        
        # 영업이익 분석
        if profit_rate > 10:
            insights.append({
                'type': 'positive',
                'title': '💰 수익성 개선',
                'content': f"영업이익이 전기 대비 {profit_rate:.1f}% 증가했습니다. 비용 효율화 또는 매출 증가의 효과로 보입니다."
            })
        elif profit_rate < -10:
            insights.append({
                'type': 'negative',
                'title': '⚠️ 수익성 악화',
                'content': f"영업이익이 전기 대비 {abs(profit_rate):.1f}% 감소했습니다. 비용 구조 점검이 필요합니다."
            })
        
        # 2. 카테고리별 비용 분석
        category_changes = []
        for category, cols in cost_categories.items():
            target_sum = sum(df_target[col].sum() for col in cols if col in df_target.columns)
            comp_sum = sum(df_comparison[col].sum() for col in cols if col in df_comparison.columns)
            diff = target_sum - comp_sum
            rate = (diff / comp_sum * 100) if comp_sum != 0 else 0
            category_changes.append({
                'category': category,
                'target': target_sum,
                'comp': comp_sum,
                'diff': diff,
                'rate': rate
            })
        
        # 가장 많이 증가한 카테고리
        df_cat = pd.DataFrame(category_changes)
        df_cat_sorted = df_cat.sort_values('diff', ascending=False)
        
        top_increase = df_cat_sorted.head(1).iloc[0] if len(df_cat_sorted) > 0 else None
        top_decrease = df_cat_sorted.tail(1).iloc[0] if len(df_cat_sorted) > 0 else None
        
        if top_increase is not None and top_increase['diff'] > 0:
            insights.append({
                'type': 'negative',
                'title': f"🔺 비용 증가 주요 항목: {top_increase['category']}",
                'content': f"{top_increase['category']} 비용이 {format_currency(top_increase['diff'], display_unit, display_divisor)} 증가했습니다 ({top_increase['rate']:.1f}% ↑). 해당 카테고리의 세부 항목을 점검하세요."
            })
        
        if top_decrease is not None and top_decrease['diff'] < 0:
            insights.append({
                'type': 'positive',
                'title': f"🔻 비용 절감 항목: {top_decrease['category']}",
                'content': f"{top_decrease['category']} 비용이 {format_currency(abs(top_decrease['diff']), display_unit, display_divisor)} 감소했습니다 ({abs(top_decrease['rate']):.1f}% ↓). 효율화가 잘 이루어지고 있습니다."
            })
        
        # 3. 개별 비용 항목 중 급증/급감 항목
        cost_analysis = analyze_cost_breakdown(df_target, df_comparison, cost_columns, display_divisor, display_unit)
        
        # 증가율 Top 3 (비교기간 금액이 일정 이상인 항목만)
        significant_costs = cost_analysis[cost_analysis['비교기간'] > 1000000]  # 100만원 이상
        if not significant_costs.empty:
            top_increase_items = significant_costs.nlargest(3, '증감률')
            for _, row in top_increase_items.iterrows():
                if row['증감률'] > 20:  # 20% 이상 증가
                    insights.append({
                        'type': 'negative',
                        'title': f"⚡ {row['비용항목']} 급증",
                        'content': f"{row['비용항목']}이(가) {row['증감률']:.1f}% 급증했습니다. 원인 파악이 필요합니다."
                    })
    
    else:
        # 비교 데이터가 없는 경우
        insights.append({
            'type': 'neutral',
            'title': '📊 현재 기간 분석',
            'content': f"총 매출액 {format_currency(total_revenue_target, display_unit, display_divisor)}, 영업이익 {format_currency(operating_profit_target, display_unit, display_divisor)}입니다."
        })
        
        # 비용 비중 분석
        total_cost = df_target['총비용'].sum() if '총비용' in df_target.columns else 0
        if total_cost > 0:
            profit_margin = (operating_profit_target / total_revenue_target * 100) if total_revenue_target != 0 else 0
            insights.append({
                'type': 'positive' if profit_margin > 10 else 'negative',
                'title': '📈 영업이익률',
                'content': f"현재 영업이익률은 {profit_margin:.1f}%입니다. {'양호한 수준입니다.' if profit_margin > 10 else '개선이 필요합니다.'}"
            })
    
    return insights


# --- NEW: 히트맵 생성 함수 ---
def create_heatmap(df, grouping_column, value_column, display_divisor, display_unit):
    """캠퍼스/브랜드 × 월별 히트맵 생성"""
    
    # 피벗 테이블 생성
    pivot_df = df.pivot_table(
        values=value_column,
        index=grouping_column,
        columns='월',
        aggfunc='sum',
        fill_value=0
    )
    
    # 월 순서 정렬
    month_order = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
    existing_months = [m for m in month_order if m in pivot_df.columns]
    pivot_df = pivot_df[existing_months]
    
    # 컬럼명 변경 (01 -> 1월)
    pivot_df.columns = [f"{int(m)}월" for m in pivot_df.columns]
    
    # 단위 적용
    pivot_df_scaled = pivot_df / display_divisor
    
    # 히트맵 생성
    fig = go.Figure(data=go.Heatmap(
        z=pivot_df_scaled.values,
        x=pivot_df_scaled.columns,
        y=pivot_df_scaled.index,
        colorscale='Blues',
        hoverongaps=False,
        hovertemplate=f'%{{y}}<br>%{{x}}: %{{z:,.1f}}{display_unit}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f'{grouping_column}별 × 월별 {value_column} 히트맵',
        xaxis_title='월',
        yaxis_title=grouping_column,
        height=max(400, len(pivot_df) * 30)  # 행 수에 따라 높이 조정
    )
    
    return fig, pivot_df_scaled


# --- 3. Session State 및 페이지 전환 로직 ---

# Session State 초기화
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'page' not in st.session_state:
    st.session_state.page = 'upload' # 'upload' or 'analysis'

# 앱 상태 초기화 및 페이지 전환 함수 (홈 버튼 역할)
def reset_app():
    st.session_state.uploaded_file = None
    st.session_state.page = 'upload'
    st.cache_data.clear() # 캐시 데이터도 함께 초기화
    st.rerun()

# --- 4. 메인 페이지 렌더링 (업로드 또는 분석) ---

# --- Year-Over-Year Quarterly Plotting Function ---
def plot_quarterly_yoy_revenue(df, target_years, comp_year, selected_months_str, is_cumulative, display_divisor, display_unit):
    
    st.markdown(f"### 📈 분기별 매출액 추이 그래프 (Year-Over-Year 비교, 단위: {display_unit})")

    # Filter only relevant months for calculation consistency
    max_month_str = max(selected_months_str) if selected_months_str else '12'
    all_months_in_range = sorted([m for m in df['월'].unique().tolist() if m <= max_month_str])
    
    # Identify all years to be plotted: Target years + Comparison year (if active)
    years_to_plot = [y for y in target_years]
    if comp_year != '선택 안함':
        years_to_plot.append(comp_year)
    years_to_plot = sorted(list(set(years_to_plot))) # Ensure unique and sorted

    yoy_plot_data = []
    
    # Quarters order for plotting
    quarter_order = ['Q1', 'Q2', 'Q3', 'Q4']

    for year in years_to_plot:
        df_year = df[
            (df['년'] == year) & 
            df['월'].isin(all_months_in_range)
        ].copy()

        if not df_year.empty:
            # Aggregate by Quarter (Group by Quarter only)
            df_agg = df_year.groupby('분기').agg(
                {'영업이익': 'sum', '매출액': 'sum'}
            ).reset_index().rename(columns={'분기': 'Quarter'})
            
            # Apply cumulative logic if needed (within the year)
            if is_cumulative:
                df_agg['Quarter_Sort'] = df_agg['Quarter'].str.replace('Q', '').astype(int)
                df_agg = df_agg.sort_values('Quarter_Sort')
                df_agg['매출액'] = df_agg['매출액'].cumsum()
                df_agg['영업이익'] = df_agg['영업이익'].cumsum()
                df_agg.drop(columns=['Quarter_Sort'], inplace=True)
            else:
                # Ensure Q1-Q4 order for non-cumulative as well
                df_agg['Quarter_Sort'] = df_agg['Quarter'].str.replace('Q', '').astype(int)
                df_agg = df_agg.sort_values('Quarter_Sort')
                df_agg.drop(columns=['Quarter_Sort'], inplace=True)
            
            df_agg['Year'] = year
            df_agg['매출액_Scaled'] = df_agg['매출액'] / display_divisor
            
            # Labeling for comparison
            if year == comp_year:
                 df_agg['Period'] = f'비교기간 ({year}년)'
            else:
                 df_agg['Period'] = f'주요기간 ({year}년)'
                 
            yoy_plot_data.append(df_agg)

    if not yoy_plot_data:
        st.warning("분기별 Yo-Y 그래프를 그릴 데이터가 없습니다.")
        return
        
    df_plot_combined = pd.concat(yoy_plot_data, ignore_index=True)
    
    fig = go.Figure()
    
    # 1. 'Quarter' (Q1, Q2, Q3, Q4)를 X축으로 사용하며, 'Period'에 따라 라인을 분리합니다.
    for period in df_plot_combined['Period'].unique():
        df_sub = df_plot_combined[df_plot_combined['Period'] == period]
        
        # '비교기간'을 검은색 점선으로, 나머지는 파란색 실선으로 설정
        is_comp_line = '비교기간' in period 

        line_color = 'black' if is_comp_line else 'blue'
        line_dash = 'dash' if is_comp_line else 'solid'
        line_width = 2 if is_comp_line else 3
        
        fig.add_trace(go.Scatter(
            # X축: Q1, Q2, Q3, Q4
            x=df_sub['Quarter'],
            y=df_sub['매출액_Scaled'],
            mode='lines+markers',
            name=period,
            line=dict(color=line_color, width=line_width, dash=line_dash),
            marker=dict(size=8, symbol='circle', line=dict(width=1, color='DarkSlateGrey'))
        ))

    # 2. Layout Updates
    mode_label = f"{' (누적)' if is_cumulative else ''}"
    fig.update_layout(
        title=f'분기별 매출액 추이{mode_label}',
        xaxis_title='분기',
        yaxis_title=f"매출액 ({display_unit})",
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )

    # 3. X-Axis Ticks: Ensure Q1, Q2, Q3, Q4 are explicitly shown
    fig.update_xaxes(
        categoryorder='array', 
        categoryarray=quarter_order, # X축 순서 강제
        automargin=True,
        showgrid=True, 
        gridcolor='#f0f0f0'
    )
    
    st.plotly_chart(fig, use_container_width=True)
# --- End of Year-Over-Year Quarterly Plotting Function ---


if st.session_state.page == 'upload':
    # --- 4-1. 파일 업로드 페이지 (메인 화면) ---
    inject_custom_css() # CSS 주입 (업로드 화면육
    st.title("매출/비용 분석 대시보드_플로우교육")
    st.markdown("---")
    
    # 중앙에 파일 업로더 배치
    col_a, col_b, col_c = st.columns([1, 2, 1])
    
    with col_b:
        st.subheader("📁파일 업로드(.xlsx)")
        
        current_uploaded_file = st.file_uploader(
            " ", 
            type=["xlsx"], 
            key="main_uploader_on_load"
        )
        
        if current_uploaded_file:
            # 파일이 새로 업로드되면 세션 상태 업데이트 및 분석 페이지로 전환
            st.session_state.uploaded_file = current_uploaded_file
            st.session_state.page = 'analysis'
            st.rerun()

    # 사이드바는 이 페이지에서는 비워둡니다.
    # st.sidebar에 아무것도 넣지 않으면 Streamlit이 자동으로 숨기거나 비웁니다.
    
else:
    # --- 4-2. 분석 대시보드 페이지 ---
    inject_custom_css() # CSS 주입 (분석 화면)
    uploaded_file = st.session_state.uploaded_file
    
    # 5. 데이터 로드 및 전처리 실행
    df = load_data(uploaded_file)
    
    if df is None:
        # 데이터 로드 실패 시 업로드 페이지로 리셋
        st.error("데이터 로드 중 오류가 발생했습니다. 파일을 다시 확인해주세요.")
        reset_app()
        
    # 데이터 로드가 성공했을 때만 분석 대시보드 표시
    else:
        data_loaded_successfully = True

        # --- 6. 사이드바 메뉴 및 필터링 로직 ---    
        
        # 요청하신 '홈 (재업로드)' 버튼
        if st.sidebar.button("main page", key="reset_button", help="클릭 시 파일 업로드 화면으로 돌아갑니다."):
            reset_app()
        
        st.sidebar.markdown("---")
        st.sidebar.header("필터 옵션")

        # 6-1. 메인 분석 메뉴 (네비게이션)
        analysis_menu = st.sidebar.radio(
            "분석 기준 선택:",
            options=["수익코드", "사업부", "브랜드", "캠퍼스"],
            key="analysis_menu"
        )
        
        # 6-2. 범용 '년'과 '월' 필터 (대상)
        st.sidebar.markdown("---")
        st.sidebar.subheader("🎯 주요 기간 필터")
        
        all_years = sorted(df['년'].unique().tolist(), reverse=True)
        selected_years = st.sidebar.multiselect(
            "년도(Year) 선택:",
            options=all_years,
            default=[]  # 초기값: 선택 안함
        )
        
        all_months_two_digits = sorted(df['월'].unique().tolist())
        display_months = [m.lstrip('0') for m in all_months_two_digits] 
        
        selected_display_months = st.sidebar.multiselect(
            "월(Month) 선택:",
            options=display_months,
            default=[]  # 초기값: 선택 안함
        )
        
        month_map = {m.lstrip('0'): m for m in all_months_two_digits}
        selected_months = [month_map[m] for m in selected_display_months]

        # 6-3. 분석 기준별 동적 필터
        st.sidebar.markdown("---")
        st.sidebar.subheader("🔍 상세 필터")
        
        # 분석 메뉴에 따른 동적 필터 생성
        selected_filter_values = {}
        
        if analysis_menu == "수익코드":
            all_revenue_codes = sorted(df['수익코드'].unique().tolist())
            selected_filter_values['수익코드'] = st.sidebar.multiselect(
                "수익코드 선택:",
                options=all_revenue_codes,
                default=[],  # 초기값: 선택 안함
                key="filter_revenue_code"
            )
        
        elif analysis_menu in ["사업부", "브랜드", "캠퍼스"]:
            # 사업부, 브랜드, 캠퍼스 분석 시 3개 필터 모두 제공
            all_business_units = sorted(df['사업부'].unique().tolist())
            selected_filter_values['사업부'] = st.sidebar.multiselect(
                "사업부 선택:",
                options=all_business_units,
                default=[],  # 초기값: 선택 안함
                key="filter_business_unit"
            )
            
            all_brands = sorted(df['브랜드'].unique().tolist())
            selected_filter_values['브랜드'] = st.sidebar.multiselect(
                "브랜드 선택:",
                options=all_brands,
                default=[],  # 초기값: 선택 안함
                key="filter_brand"
            )
            
            all_campuses = sorted(df['캠퍼스'].unique().tolist())
            selected_filter_values['캠퍼스'] = st.sidebar.multiselect(
                "캠퍼스 선택:",
                options=all_campuses,
                default=[],  # 초기값: 선택 안함
                key="filter_campus"
            )
        
        # 6-4. 시간 집계 옵션
        st.sidebar.markdown("---")
        st.sidebar.subheader("📈 시간 추이 분석 옵션")
        
        time_agg_type = st.sidebar.selectbox(
            "추이 분석 단위:",
            options=["월별", "분기별"],
            key="time_agg_type"
        )
        is_cumulative = st.sidebar.checkbox("누적 합계 보기", key="is_cumulative", value=False)
        
        # 6-5. 단위 설정 필터
        st.sidebar.markdown("---")
        st.sidebar.subheader("💰 표시 단위 설정")

        unit_options = {
            "원 (W)": (1, " 원"),
            "천 원 (K)": (1000, " 천 원"),
            "백만 원 (M)": (1000000, " 백만 원"),
            "천만 원 (10M)": (10000000, " 천만 원"),
            "억 원 (B)": (100000000, " 억 원")
        }

        selected_unit_label = st.sidebar.selectbox(
            "단위 선택:",
            options=list(unit_options.keys()),
            index=2, # 기본값: 백만 원
            key="display_unit_selector"
        )

        display_divisor, display_unit = unit_options[selected_unit_label]

        # --- 7. 메인 화면 비교 기간 설정 및 데이터 필터링 ---
        
        st.title("매출/비용 분석 대시보드")
        
        col_comp_year, col_comp_month = st.columns(2)
        
        with col_comp_year:
            comparison_year = st.selectbox(
                "비교 년도(Year) 선택:",
                options=['선택 안함'] + all_years,
                index=0,
                key="comparison_year_selector",
                help="비교 기준으로 사용할 년도를 선택합니다."
            )
        
        with col_comp_month:
            comparison_selected_display_months = st.multiselect(
                "비교 월(Month) 선택:",
                options=display_months,
                default=[],  # 초기값: 선택 안함
                key="comparison_month_selector",
                help="비교 년도 내에서 주요 기간과 비교할 월을 선택합니다."
            )
            comparison_selected_months = [month_map[m] for m in comparison_selected_display_months]
            
        is_comparison_active = comparison_year != '선택 안함'
        st.markdown("---")

        # --- 데이터 필터링 로직 ---

        # 1. Target Data Filtering (Metrics/Breakdown - 사용자가 선택한 월만 합산)
        df_target = pd.DataFrame()
        if selected_years and selected_months:
            df_target = df[
                df['년'].isin(selected_years) & 
                df['월'].isin(selected_months)
            ].copy()
            
            # 동적 필터 적용
            for filter_col, filter_values in selected_filter_values.items():
                if filter_values:  # 선택된 값이 있을 때만
                    df_target = df_target[df_target[filter_col].isin(filter_values)]
            
        # 2. Trend Data Filtering (Graph/Trend Table - 1월부터 선택된 마지막 월까지 모두 포함)
        df_trend_base = pd.DataFrame()
        df_comp_trend_base = pd.DataFrame()
        
        if selected_years and selected_months:
            # 주요 기간: 1월부터 선택된 가장 큰 월까지 포함 (그래프 연속성 유지용)
            max_selected_month_str = max(selected_months)
            all_months_in_range_target = sorted([m for m in all_months_two_digits if m <= max_selected_month_str])

            df_trend_base = df[
                df['년'].isin(selected_years) & 
                df['월'].isin(all_months_in_range_target)
            ].copy()
            
            # 동적 필터 적용
            for filter_col, filter_values in selected_filter_values.items():
                if filter_values:
                    df_trend_base = df_trend_base[df_trend_base[filter_col].isin(filter_values)]
            
            # Comparison Data Filtering: Metrics/Breakdown (사용자가 선택한 월만 합산)
            df_comparison = pd.DataFrame()
            if is_comparison_active and comparison_selected_months:
                df_comparison = df[
                    (df['년'] == comparison_year) & 
                    df['월'].isin(comparison_selected_months)
                ].copy()
                
                # 동적 필터 적용
                for filter_col, filter_values in selected_filter_values.items():
                    if filter_values:
                        df_comparison = df_comparison[df_comparison[filter_col].isin(filter_values)]

                # 비교 기간: 1월부터 선택된 가장 큰 월까지 포함 (그래프 연속성 유지용)
                max_comp_month_str = max(comparison_selected_months)
                all_months_in_range_comp = sorted([m for m in all_months_two_digits if m <= max_comp_month_str])
                
                df_comp_trend_base = df[
                    (df['년'] == comparison_year) & 
                    df['월'].isin(all_months_in_range_comp)
                ].copy()
                
                # 동적 필터 적용
                for filter_col, filter_values in selected_filter_values.items():
                    if filter_values:
                        df_comp_trend_base = df_comp_trend_base[df_comp_trend_base[filter_col].isin(filter_values)]

        # 조건부 필터링 및 분석 기준 설정 (current_df는 Metrics/Breakdown에만 사용)
        current_df = df_target.copy()
        grouping_column_map = {
            "수익코드": '수익코드',
            "사업부": '사업부',
            "브랜드": '브랜드',
            "캠퍼스": '캠퍼스'
        }
        grouping_column = grouping_column_map.get(analysis_menu, '수익코드') 
        
        if analysis_menu == "수익코드":
            breakdown_cols = ['수익코드']
        elif analysis_menu == "사업부":
            breakdown_cols = ['사업부', '브랜드', '캠퍼스']
        elif analysis_menu == "브랜드":
            breakdown_cols = ['브랜드', '캠퍼스']
        elif analysis_menu == "캠퍼스":
            breakdown_cols = ['캠퍼스']
        else:
            breakdown_cols = ['수익코드']
        
        # --- 8. 분석 결과 표시 (필터링된 데이터 기반) ---
        
        if current_df.empty:
            st.warning("선택하신 필터 조건에 해당하는 데이터가 없습니다. 필터를 조정해 보세요.")
        else:
            
            # 8-1. 주요 지표 요약 (메트릭)
            total_revenue_target = current_df['매출액'].sum()
            total_cost_target = current_df['총비용'].sum()
            operating_profit_target = current_df['영업이익'].sum()
            
            delta_revenue, delta_cost, delta_profit = 0, 0, 0  # 초기값 0으로 설정
            delta_label = ""
            
            # 비교 데이터 초기화
            total_revenue_comp = 0
            total_cost_comp = 0
            operating_profit_comp = 0
            
            # Delta 값 및 Delta HTML 초기화
            delta_profit_html = ""
            delta_revenue_html = ""
            delta_cost_html = ""
            
            # AI 인사이트 초기화
            insights = []
            
            if is_comparison_active and not df_comparison.empty:
                total_revenue_comp = df_comparison['매출액'].sum()
                total_cost_comp = df_comparison['총비용'].sum()
                operating_profit_comp = df_comparison['영업이익'].sum()
                
                delta_revenue = total_revenue_target - total_revenue_comp
                delta_cost = total_cost_target - total_cost_comp
                delta_profit = operating_profit_target - operating_profit_comp
                
                # 증감률 계산
                profit_rate = (delta_profit / operating_profit_comp * 100) if operating_profit_comp != 0 else 0
                revenue_rate = (delta_revenue / total_revenue_comp * 100) if total_revenue_comp != 0 else 0
                cost_rate = (delta_cost / total_cost_comp * 100) if total_cost_comp != 0 else 0
                
                # 증감액 스케일링
                delta_profit_scaled = delta_profit / display_divisor
                delta_revenue_scaled = delta_revenue / display_divisor
                delta_cost_scaled = delta_cost / display_divisor
                
                # Delta HTML 생성 (색상 포함)
                # 영업이익
                profit_color = "#2563eb" if delta_profit >= 0 else "#ef4444"
                profit_symbol = "+" if delta_profit >= 0 else "△"
                profit_rate_str = f"{profit_rate:.1f}%" if profit_rate >= 0 else f"{abs(profit_rate):.1f}%"
                delta_profit_html = f'<span style="color: {profit_color}; font-weight: 600; font-size: 0.95rem;">{profit_symbol}{abs(delta_profit_scaled):.1f}{display_unit} ({profit_symbol}{profit_rate_str})</span>'
                
                # 매출액
                revenue_color = "#2563eb" if delta_revenue >= 0 else "#ef4444"
                revenue_symbol = "+" if delta_revenue >= 0 else "△"
                revenue_rate_str = f"{revenue_rate:.1f}%" if revenue_rate >= 0 else f"{abs(revenue_rate):.1f}%"
                delta_revenue_html = f'<span style="color: {revenue_color}; font-weight: 600; font-size: 0.95rem;">{revenue_symbol}{abs(delta_revenue_scaled):.1f}{display_unit} ({revenue_symbol}{revenue_rate_str})</span>'
                
                # 총비용 (비용은 감소가 좋으므로 색상 반대)
                cost_color = "#ef4444" if delta_cost >= 0 else "#2563eb"
                cost_symbol = "+" if delta_cost >= 0 else "△"
                cost_rate_str = f"{cost_rate:.1f}%" if cost_rate >= 0 else f"{abs(cost_rate):.1f}%"
                delta_cost_html = f'<span style="color: {cost_color}; font-weight: 600; font-size: 0.95rem;">{cost_symbol}{abs(delta_cost_scaled):.1f}{display_unit} ({cost_symbol}{cost_rate_str})</span>'
                
                comp_months_display = ', '.join([m.lstrip('0') for m in comparison_selected_months])
                delta_label = f" vs. {comparison_year}년 ({comp_months_display}월)"
                
            else:
                delta_label = " (비교 기간 미선택)"

            target_years_display = ', '.join(selected_years)
            target_months_display = ', '.join([m.lstrip('0') for m in selected_months])
            target_label = f"{target_years_display}년 ({target_months_display}월) 합계"

            st.markdown(f"### 🔑 핵심 손익 지표 (누적 합계){delta_label}")
            
            col_profit, col_revenue, col_cost = st.columns(3)
            
            # 영업 이익 Metric (1순위) - 커스텀 HTML 카드
            with col_profit:
                st.markdown(f"""
                <div style="background: white; border-radius: 16px; padding: 1.75rem 1.5rem; 
                            box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1); border-left: 4px solid #3b82f6;">
                    <div style="font-size: 0.875rem; font-weight: 600; color: #64748b; 
                                text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem;">
                        영업 이익 ({target_label})
                    </div>
                    <div style="font-size: 2rem; font-weight: 700; color: #0f172a; margin-bottom: 0.5rem;">
                        {format_currency(operating_profit_target, display_unit, display_divisor)}
                    </div>
                    <div>
                        {delta_profit_html}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # 매출액 Metric (2순위) - 커스텀 HTML 카드
            with col_revenue:
                st.markdown(f"""
                <div style="background: white; border-radius: 16px; padding: 1.75rem 1.5rem; 
                            box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1); border-left: 4px solid #3b82f6;">
                    <div style="font-size: 0.875rem; font-weight: 600; color: #64748b; 
                                text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem;">
                        총 매출액 ({target_label})
                    </div>
                    <div style="font-size: 2rem; font-weight: 700; color: #0f172a; margin-bottom: 0.5rem;">
                        {format_currency(total_revenue_target, display_unit, display_divisor)}
                    </div>
                    <div>
                        {delta_revenue_html}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # 총 비용 Metric (3순위) - 커스텀 HTML 카드
            with col_cost:
                st.markdown(f"""
                <div style="background: white; border-radius: 16px; padding: 1.75rem 1.5rem; 
                            box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1); border-left: 4px solid #3b82f6;">
                    <div style="font-size: 0.875rem; font-weight: 600; color: #64748b; 
                                text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem;">
                        총 비용 ({target_label})
                    </div>
                    <div style="font-size: 2rem; font-weight: 700; color: #0f172a; margin-bottom: 0.5rem;">
                        {format_currency(total_cost_target, display_unit, display_divisor)}
                    </div>
                    <div>
                        {delta_cost_html}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")
            
            # ================================================================
            # --- NEW: 8-1-1. AI 인사이트 분석 섹션 ---
            # ================================================================
            st.markdown("### 🤖 AI 인사이트 분석")
            
            with st.expander("AI 분석 결과 보기", expanded=True):
                insights = generate_ai_insights(
                    df_target=current_df,
                    df_comparison=df_comparison if is_comparison_active else None,
                    cost_columns=COST_COLUMNS,
                    cost_categories=COST_CATEGORIES,
                    total_revenue_target=total_revenue_target,
                    total_revenue_comp=total_revenue_comp,
                    operating_profit_target=operating_profit_target,
                    operating_profit_comp=operating_profit_comp,
                    display_divisor=display_divisor,
                    display_unit=display_unit
                )
                
                if insights:
                    for insight in insights:
                        icon_map = {'positive': '✅', 'negative': '⚠️', 'neutral': 'ℹ️'}
                        color_map = {'positive': '#d1fae5', 'negative': '#fee2e2', 'neutral': '#e5e7eb'}
                        border_map = {'positive': '#10b981', 'negative': '#ef4444', 'neutral': '#6b7280'}
                        
                        st.markdown(f"""
                        <div style="background-color: {color_map[insight['type']]}; 
                                    padding: 1rem; 
                                    border-radius: 0.5rem; 
                                    margin: 0.5rem 0;
                                    border-left: 4px solid {border_map[insight['type']]};">
                            <strong>{icon_map[insight['type']]} {insight['title']}</strong><br>
                            {insight['content']}
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("분석할 데이터가 충분하지 않습니다.")
            
            st.markdown("---")
            
            # --- 8-2. 상세 Breakdown 테이블 (영업이익, 매출액) ---
            st.markdown(f"### 📊 상세 손익 내역 (단위: {selected_unit_label})")
            
            df_target_group = current_df.groupby(breakdown_cols)[['매출액', '영업이익']].sum().reset_index()
            df_target_group.columns = breakdown_cols + ['매출액', '영업이익'] 
            df_merged = df_target_group.copy()
            
            if is_comparison_active and not df_comparison.empty:
                df_comp_group = df_comparison.groupby(breakdown_cols)[['매출액', '영업이익']].sum().reset_index()
                df_comp_group.columns = breakdown_cols + ['비교 매출액', '비교 영업이익']
                df_merged = pd.merge(df_target_group, df_comp_group, on=breakdown_cols, how='outer').fillna(0)
                
                df_merged['매출액 증감'] = df_merged['매출액'] - df_merged['비교 매출액']
                df_merged['영업이익 증감'] = df_merged['영업이익'] - df_merged['비교 영업이익']
                
                df_merged['매출액 증감률'] = np.where(df_merged['비교 매출액'] == 0, np.where(df_merged['매출액'] == 0, 0, np.inf), (df_merged['매출액 증감'] / df_merged['비교 매출액']) * 100)
                df_merged['영업이익 증감률'] = np.where(df_merged['비교 영업이익'] == 0, np.where(df_merged['영업이익'] == 0, 0, np.inf), (df_merged['영업이익 증감'] / df_merged['비교 영업이익']) * 100)
                
                final_cols = breakdown_cols + ['영업이익', '영업이익 증감', '영업이익 증감률', '매출액', '매출액 증감', '매출액 증감률']
                df_display_raw = df_merged[final_cols].copy()
            else:
                df_display_raw = df_merged.copy()


            # Grand Total Calculation
            total_row_data = df_display_raw.select_dtypes(include=np.number).sum().to_dict()
            
            if is_comparison_active:
                total_comp_profit = total_row_data['영업이익'] - total_row_data['영업이익 증감']
                total_comp_revenue = total_row_data['매출액'] - total_row_data['매출액 증감']
                
                total_profit_rate = np.where(total_comp_profit == 0, np.where(total_row_data['영업이익'] == 0, 0, np.inf), (total_row_data['영업이익 증감'] / total_comp_profit) * 100)
                total_revenue_rate = np.where(total_comp_revenue == 0, np.where(total_row_data['매출액'] == 0, 0, np.inf), (total_row_data['매출액 증감'] / total_comp_revenue) * 100)
                
                total_row_data['영업이익 증감률'] = total_profit_rate
                total_row_data['매출액 증감률'] = total_revenue_rate


            # 데이터 포맷팅
            df_display = df_display_raw.copy()
            currency_cols = [col for col in df_display.columns if '영업이익' in col or '매출액' in col]
            rate_cols = [col for col in df_display.columns if '증감률' in col]
            
            for col in currency_cols:
                if col in rate_cols: continue
                df_display[col] = df_display[col].apply(
                    lambda x: format_currency(x, display_unit, display_divisor)
                )

            for col in rate_cols:
                df_display[col] = df_display[col].apply(
                    lambda x: f"{x:,.1f} %" if not pd.isna(x) and x != np.inf else ('N/A' if x == 0 else 'Inf %')
                )

            # 총합계 행 포맷팅 및 추가
            total_display_row = {
                breakdown_cols[0]: '총합계', 
            }
            if len(breakdown_cols) > 1:
                for col in breakdown_cols[1:]:
                    total_display_row[col] = ''
            
            for col, val in total_row_data.items():
                if '증감률' in col:
                    total_display_row[col] = f"{val:,.1f} %" if not pd.isna(val) and val != np.inf else ('N/A' if val == 0 else 'Inf %')
                elif '영업이익' in col or '매출액' in col:
                    total_display_row[col] = format_currency(val, display_unit, display_divisor)
                
            df_total = pd.DataFrame([total_display_row])
            df_display = pd.concat([df_display, df_total], ignore_index=True)

            st.dataframe(df_display, use_container_width=True)
            
            # 엑셀 다운로드용 데이터 저장 (포맷팅 전 raw 데이터)
            df_summary_for_export = df_display_raw.copy()
            
            st.markdown("---")
            
            # ================================================================
            # --- NEW: 8-2-1. 비용 항목별 상세 분석 섹션 ---
            # ================================================================
            st.markdown(f"### 💸 비용 항목별 상세 분석 (단위: {selected_unit_label})")
            
            # 비용 컬럼이 데이터에 존재하는지 확인
            existing_cost_cols = [col for col in COST_COLUMNS if col in current_df.columns]
            
            if existing_cost_cols:
                with st.expander("비용 항목별 증감 분석 보기", expanded=False):
                    
                    cost_df = analyze_cost_breakdown(
                        df_target=current_df,
                        df_comparison=df_comparison if is_comparison_active else None,
                        cost_columns=existing_cost_cols,
                        display_divisor=display_divisor,
                        display_unit=display_unit
                    )
                    
                    # Top 5 증가/감소 항목 시각화
                    col_inc, col_dec = st.columns(2)
                    
                    with col_inc:
                        st.markdown("#### 🔺 비용 증가 Top 5")
                        top_increase = cost_df.nlargest(5, '증감액')
                        top_increase_positive = top_increase[top_increase['증감액'] > 0]
                        
                        if not top_increase_positive.empty:
                            fig_inc = px.bar(
                                top_increase_positive,
                                x='증감액',
                                y='비용항목',
                                orientation='h',
                                color_discrete_sequence=['#ef4444']
                            )
                            fig_inc.update_layout(
                                xaxis_title=f"증감액 ({display_unit})",
                                yaxis_title="",
                                height=300,
                                showlegend=False
                            )
                            fig_inc.update_traces(
                                text=[format_currency(x, display_unit, display_divisor) for x in top_increase_positive['증감액']],
                                textposition='outside'
                            )
                            st.plotly_chart(fig_inc, use_container_width=True)
                        else:
                            st.info("증가한 비용 항목이 없습니다.")
                    
                    with col_dec:
                        st.markdown("#### 🔻 비용 감소 Top 5")
                        top_decrease = cost_df.nsmallest(5, '증감액')
                        top_decrease_negative = top_decrease[top_decrease['증감액'] < 0]
                        
                        if not top_decrease_negative.empty:
                            fig_dec = px.bar(
                                top_decrease_negative,
                                x='증감액',
                                y='비용항목',
                                orientation='h',
                                color_discrete_sequence=['#10b981']
                            )
                            fig_dec.update_layout(
                                xaxis_title=f"증감액 ({display_unit})",
                                yaxis_title="",
                                height=300,
                                showlegend=False
                            )
                            fig_dec.update_traces(
                                text=[format_currency(x, display_unit, display_divisor) for x in top_decrease_negative['증감액']],
                                textposition='outside'
                            )
                            st.plotly_chart(fig_dec, use_container_width=True)
                        else:
                            st.info("감소한 비용 항목이 없습니다.")
                    
                    # 전체 비용 항목 테이블
                    st.markdown("#### 📋 전체 비용 항목 상세")
                    
                    # 포맷팅
                    cost_display = cost_df.copy()
                    cost_display = cost_display.sort_values('증감액', ascending=False)
                    
                    cost_display['주요기간'] = cost_display['주요기간'].apply(
                        lambda x: format_currency(x, display_unit, display_divisor)
                    )
                    cost_display['비교기간'] = cost_display['비교기간'].apply(
                        lambda x: format_currency(x, display_unit, display_divisor)
                    )
                    cost_display['증감액'] = cost_display['증감액'].apply(
                        lambda x: format_currency(x, display_unit, display_divisor)
                    )
                    cost_display['증감률'] = cost_display['증감률'].apply(
                        lambda x: f"{x:,.1f} %" if not pd.isna(x) and x != np.inf else ('N/A' if x == 0 else 'Inf %')
                    )
                    
                    st.dataframe(cost_display, use_container_width=True)
            else:
                st.warning("비용 항목 컬럼이 데이터에 없어 비용 분석을 수행할 수 없습니다.")
            
            st.markdown("---")
            
            # ================================================================
            # --- NEW: 8-2-2. 히트맵 섹션 ---
            # ================================================================
            st.markdown(f"### 🗺️ 히트맵 분석")
            
            with st.expander("히트맵 보기", expanded=False):
                
                heatmap_col1, heatmap_col2 = st.columns(2)
                
                with heatmap_col1:
                    heatmap_grouping = st.selectbox(
                        "히트맵 기준 선택:",
                        options=['캠퍼스', '브랜드', '사업부', '수익코드'],
                        key="heatmap_grouping"
                    )
                
                with heatmap_col2:
                    heatmap_value = st.selectbox(
                        "표시 지표 선택:",
                        options=['매출액', '영업이익'],
                        key="heatmap_value"
                    )
                
                # 히트맵 생성
                if not current_df.empty:
                    fig_heatmap, pivot_data = create_heatmap(
                        df=current_df,
                        grouping_column=heatmap_grouping,
                        value_column=heatmap_value,
                        display_divisor=display_divisor,
                        display_unit=display_unit
                    )
                    
                    st.plotly_chart(fig_heatmap, use_container_width=True)
                    
                    # 히트맵 데이터 테이블
                    st.markdown("#### 📋 히트맵 데이터 (상세)")
                    
                    # 포맷팅
                    pivot_display = pivot_data.copy()
                    for col in pivot_display.columns:
                        pivot_display[col] = pivot_display[col].apply(
                            lambda x: f"{x:,.1f}"
                        )
                    
                    st.dataframe(pivot_display, use_container_width=True)
                else:
                    st.warning("히트맵을 생성할 데이터가 없습니다.")
            
            st.markdown("---")

            # --- 8-3. 기간별 추이 분석 테이블 ---
            
            time_col = '년월' if time_agg_type == "월별" else '년분기'
            sort_col = 'sort_key'

            target_label_full = f'주요기간 ({", ".join(selected_years)}년)' 
            comp_label_full = f'비교기간 ({comparison_year}년)'
            
            df_trend_target = aggregate_profit_trend(df_trend_base, time_col, sort_col, is_cumulative, target_label_full)
            
            df_trend_comp = None
            if is_comparison_active and not df_comp_trend_base.empty:
                df_trend_comp = aggregate_profit_trend(df_comp_trend_base, time_col, sort_col, is_cumulative, comp_label_full)
            
            # 추이 테이블 생성 및 표시
            mode_label = f"{time_agg_type}{' (누적)' if is_cumulative else ''}"
            st.markdown(f"### 📋 기간별 손익 추이 테이블 ({mode_label}, 단위: {selected_unit_label})")
            
            if df_trend_target is not None:
                
                if df_trend_comp is not None:
                    df_trend_target.rename(columns={'영업이익': '영업이익', '매출액': '매출액'}, inplace=True)
                    df_trend_comp.rename(columns={'영업이익': '비교 영업이익', '매출액': '비교 매출액'}, inplace=True)
                    
                    df_trend_merged = pd.merge(
                        df_trend_target.drop(columns=['기간']), 
                        df_trend_comp.drop(columns=['기간']), 
                        on='display_label', 
                        how='outer'
                    ).fillna(0)
                    
                    df_trend_merged['영업이익 증감'] = df_trend_merged['영업이익'] - df_trend_merged['비교 영업이익']
                    df_trend_merged['매출액 증감'] = df_trend_merged['매출액'] - df_trend_merged['비교 매출액']
                    
                    df_trend_merged['영업이익 증감률'] = np.where(df_trend_merged['비교 영업이익'] == 0, np.where(df_trend_merged['영업이익'] == 0, 0, np.inf), (df_trend_merged['영업이익 증감'] / df_trend_merged['비교 영업이익']) * 100)
                    df_trend_merged['매출액 증감률'] = np.where(df_trend_merged['비교 매출액'] == 0, np.where(df_trend_merged['매출액'] == 0, 0, np.inf), (df_trend_merged['매출액 증감'] / df_trend_merged['비교 매출액']) * 100)
                    
                    df_trend_raw = df_trend_merged[[
                        'display_label', 
                        '영업이익', '영업이익 증감', '영업이익 증감률',
                        '매출액', '매출액 증감', '매출액 증감률',
                    ]].copy()
                    
                    label_header = '월' if time_agg_type == '월별' else '년분기'
                    df_trend_raw.rename(columns={'display_label': label_header}, inplace=True)
                    
                else:
                    df_trend_target.rename(columns={'영업이익': '영업이익', '매출액': '매출액'}, inplace=True)
                    df_trend_raw = df_trend_target.drop(columns=['기간']).copy()
                    label_header = '월' if time_agg_type == '월별' else '년분기'
                    df_trend_raw.rename(columns={'display_label': label_header}, inplace=True)


                # 데이터 포맷팅 (추이 테이블)
                df_trend_display = df_trend_raw.copy()
                currency_cols_trend = [col for col in df_trend_display.columns if '영업이익' in col or '매출액' in col]
                rate_cols_trend = [col for col in df_trend_display.columns if '증감률' in col]
                
                for col in currency_cols_trend:
                    if col in rate_cols_trend: continue
                    df_trend_display[col] = df_trend_display[col].apply(
                        lambda x: format_currency(x, display_unit, display_divisor)
                    )

                for col in rate_cols_trend:
                    df_trend_display[col] = df_trend_display[col].apply(
                        lambda x: f"{x:,.1f} %" if not pd.isna(x) and x != np.inf else ('N/A' if x == 0 else 'Inf %')
                    )

                st.dataframe(df_trend_display, use_container_width=True)
                
                # 추이 데이터 저장 (엑셀 다운로드용)
                df_trend_for_export = df_trend_raw.copy()
                
                st.markdown("---")
                
                # ================================================================
                # --- NEW: 다운로드 버튼 섹션 ---
                # ================================================================
                st.markdown("### 📥 리포트 다운로드")
                
                col_download1, col_download2 = st.columns(2)
                
                with col_download1:
                    # 엑셀 다운로드 버튼
                    try:
                        # 비용 분석 데이터도 포함 (있을 경우)
                        cost_df_for_export = None
                        if existing_cost_cols:
                            cost_df_for_export = analyze_cost_breakdown(
                                df_target=current_df,
                                df_comparison=df_comparison if is_comparison_active else None,
                                cost_columns=existing_cost_cols,
                                display_divisor=1,  # Raw 값 사용
                                display_unit=" 원"
                            )
                        
                        excel_file = create_excel_report(
                            df_summary=df_summary_for_export,
                            df_trend=df_trend_for_export,
                            df_cost_analysis=cost_df_for_export
                        )
                        
                        filename_excel = f"손익분석_{target_years_display}년_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                        
                        st.download_button(
                            label="📊 엑셀 파일 다운로드",
                            data=excel_file,
                            file_name=filename_excel,
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            help="상세 손익 내역과 추이 데이터를 엑셀 파일로 다운로드합니다.",
                            use_container_width=True
                        )
                    except Exception as e:
                        st.error(f"엑셀 파일 생성 중 오류 발생: {e}")
                
                with col_download2:
                    # HTML 리포트 다운로드 버튼
                    try:
                        html_report = create_html_report(
                            total_revenue_target=total_revenue_target,
                            total_cost_target=total_cost_target,
                            operating_profit_target=operating_profit_target,
                            total_revenue_comp=total_revenue_comp,
                            total_cost_comp=total_cost_comp,
                            operating_profit_comp=operating_profit_comp,
                            delta_revenue=delta_revenue if is_comparison_active else 0,
                            delta_cost=delta_cost if is_comparison_active else 0,
                            delta_profit=delta_profit if is_comparison_active else 0,
                            target_label=target_label,
                            comparison_year=comparison_year,
                            display_unit=display_unit,
                            display_divisor=display_divisor,
                            df_summary=df_summary_for_export,
                            df_trend=df_trend_for_export,
                            insights=insights
                        )
                        
                        filename_html = f"손익분석리포트_{target_years_display}년_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                        
                        st.download_button(
                            label="📧 HTML 리포트 다운로드",
                            data=html_report,
                            file_name=filename_html,
                            mime="text/html",
                            help="이메일 첨부용 HTML 리포트를 다운로드합니다.",
                            use_container_width=True
                        )
                    except Exception as e:
                        st.error(f"HTML 리포트 생성 중 오류 발생: {e}")
                
                st.markdown("---")
                
                # --- 8-4. 추이 그래프 (Plotly) - 월별 vs 분기별 분리 ---
                
                if time_agg_type == "월별":
                    # --- 월별: 연속적인 시간 흐름 그래프 ---
                    st.markdown(f"### 📈 월별 매출액 추이 그래프 ({mode_label}, 단위: {selected_unit_label})") 
                    
                    # 1. 그래프용 데이터 준비
                    df_plot_target = aggregate_profit_trend(df_trend_base, time_col, sort_col, is_cumulative, target_label_full)
                    df_plot_target['매출액_Scaled'] = df_plot_target['매출액'] / display_divisor
                    
                    # 선택 월 강조용 마커 (실제 선택된 월만)
                    df_plot_target_markers_raw = aggregate_profit_trend(df_target, time_col, sort_col, is_cumulative, target_label_full)
                    df_plot_target_markers = df_plot_target_markers_raw.copy()
                    df_plot_target_markers['매출액_Scaled'] = df_plot_target_markers['매출액'] / display_divisor
                    
                    
                    fig = go.Figure()

                    # A. 주요기간 (Target) 라인 Trace
                    fig.add_trace(go.Scatter(
                        x=df_plot_target['display_label'],
                        y=df_plot_target['매출액_Scaled'],
                        mode='lines',
                        name=target_label_full,
                        line=dict(color='blue', width=3)
                    ))

                    # B. 선택 월 강조 (Target Markers) Scatter Trace
                    # 선택된 월에만 마커 표시
                    fig.add_trace(go.Scatter(
                        x=df_plot_target_markers['display_label'],
                        y=df_plot_target_markers['매출액_Scaled'],
                        mode='markers',
                        name='선택 월 강조',
                        showlegend=False, # 범례에서 숨김
                        marker=dict(
                            size=10, 
                            color='blue', 
                            line=dict(width=2, color='DarkSlateGrey')
                        ),
                        hoverinfo='text',
                        text=[f"{y:,.1f} {selected_unit_label}" for y in df_plot_target_markers['매출액_Scaled'].tolist()]
                    ))

                    # C. 비교기간 (Comparison) 라인 Trace
                    if is_comparison_active and df_trend_comp is not None:
                        df_plot_comp = aggregate_profit_trend(df_comp_trend_base, time_col, sort_col, is_cumulative, comp_label_full)
                        df_plot_comp['매출액_Scaled'] = df_plot_comp['매출액'] / display_divisor
                        
                        fig.add_trace(go.Scatter(
                            x=df_plot_comp['display_label'],
                            y=df_plot_comp['매출액_Scaled'],
                            mode='lines',
                            name=comp_label_full,
                            line=dict(color='black', dash='dash', width=2) # 검은색, 점선으로 표시
                        ))

                    # 3. 레이아웃 업데이트
                    fig.update_layout(
                        title=f'{mode_label} 매출액 추이',
                        xaxis_title='월',
                        yaxis_title=f"매출액 ({selected_unit_label})",
                        hovermode="x unified",
                        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
                    )

                    # 4. X-Axis Ticks (월별 분석): 1, 2, 3... 모두 표시
                    fig.update_xaxes(
                        dtick=1, 
                        automargin=True,
                        showgrid=True, 
                        gridcolor='#f0f0f0'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)

                elif time_agg_type == "분기별":
                    # --- 분기별: Year-Over-Year 비교 그래프 ---
                    
                    # 동적 필터가 적용된 데이터로 그래프 생성
                    df_filtered_for_graph = df.copy()
                    for filter_col, filter_values in selected_filter_values.items():
                        if filter_values:
                            df_filtered_for_graph = df_filtered_for_graph[df_filtered_for_graph[filter_col].isin(filter_values)]
                    
                    # 새로운 Yo-Y 그래프 함수 호출
                    plot_quarterly_yoy_revenue(
                        df=df_filtered_for_graph, 
                        target_years=selected_years, 
                        comp_year=comparison_year if is_comparison_active else '선택 안함', 
                        selected_months_str=selected_months, 
                        is_cumulative=is_cumulative, 
                        display_divisor=display_divisor, 
                        display_unit=display_unit
                    )
                
            else:
                st.warning("기간별 추이 데이터를 생성할 수 없습니다. 주요 기간 필터를 확인해주세요.")