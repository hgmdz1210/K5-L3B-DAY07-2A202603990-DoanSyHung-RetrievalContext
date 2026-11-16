"""Benchmark thử thách nâng cao (IDR 2026): định nghĩa + kiểm chứng với bản gốc, ghi data/idr/benchmark.json.

Chạy: .venv/Scripts/python -m src.idr.benchmark
Soạn bằng cách đọc ảnh trang gốc (render từ PDF) và text layer; KHÔNG dùng model sinh đáp án.
Chia dev/test trước mọi thử nghiệm. Gold chỉ dùng để chấm, không đưa vào pipeline.

Evidence:
- `chunk`  : chunk index chứa vùng evidence (figure/bảng/hồ sơ) — dùng để đo Context Evidence Rate.
- `quote`  : đoạn text nguyên văn; script tìm chunk text chứa nó trên đúng trang.
- `values` : chuỗi phải có nguyên văn trong TEXT LAYER GỐC của trang (kiểm chứng với bản gốc, không với vision).
- `visual` : giá trị đọc bằng mắt từ hình (không có nhãn số) — đã đối chiếu thủ công với ảnh crop.
- `mode`   : 'all' = cần mọi evidence (multi-hop); 'any' = evidence thay thế.
- `alt_chunks`: chunk khác nêu cùng nội dung (evidence thay thế cho đúng mục đó).
Trang: pdf_page (1-280) và printed_page (số in trên giấy).
"""
from __future__ import annotations

import json
import re

import pymupdf

from .. import config
from .pages import OUT, PDF

T11, T12 = 'idr:p028:Table_1.1', 'idr:p063:Table_1.2'

Q = [
    # ------------------------------- DEV (15 có đáp án + 3 thiếu bằng chứng) -------------------------------
    dict(id='d01', split='dev', group='table',
         question='According to Table 1.1, what was the total external debt stock of low- and middle-income countries in 2018?',
         answer='US$7,382.3 billion.',
         evidence=[dict(chunk=T11, pdf_page=28, region='Table 1.1, row "Debt stock", column 2018', values=['7,382.3'])]),
    dict(id='d02', split='dev', group='table',
         question='According to Table 1.2, how did net debt inflows from bilateral creditors to IDA-eligible countries change between 2018 and 2025?',
         answer='They fell from US$22.1 billion in 2018 to US$1.4 billion in 2025, a decline of US$20.7 billion.',
         calc='22.1 - 1.4 = 20.7',
         evidence=[dict(chunk=T12, pdf_page=63, region='Table 1.2, row "Bilateral creditors", columns 2018 and 2025', values=['22.1', '1.4'])]),
    dict(id='d03', split='dev', group='table',
         question="In Table 1.1, by how much did LMICs' short-term external debt change from 2015 to 2016?",
         answer='It decreased from US$1,519.1 billion to US$1,475.0 billion, i.e. by US$44.1 billion.',
         calc='1,475.0 - 1,519.1 = -44.1',
         evidence=[dict(chunk=T11, pdf_page=28, region='Table 1.1, row "Short-term", columns 2015-2016', values=['1,519.1', '1,475.0'])]),
    dict(id='d04', split='dev', group='image_diagram',
         question='In the World Bank Group Debt Statistics Program three-tier architecture diagram, what label is on the dashed feedback line that loops back across the tiers?',
         answer='"Gaps and findings strengthen reporting" (the dashed loop runs from Tier 3 back to Tier 1).',
         evidence=[dict(chunk='idr:p132:Figure_3B.1', pdf_page=132, region='Figure 3B.1, dashed arrow under the three tier boxes',
                        values=['Gaps and findings strengthen reporting'])]),
    dict(id='d05', split='dev', group='image_diagram',
         question="In Kenya's country profile creditor breakdown diagram (Figure 2), which is the largest multilateral creditor and which is the largest named bilateral creditor, and how much is owed to each?",
         answer='World Bank-IDA, US$13,054 million (multilateral); China, US$4,923 million (bilateral).',
         evidence=[dict(chunk='idr:p202:profile', pdf_page=202, region='Kenya profile, Figure 2, rows MULTILATERAL and BILATERAL',
                        values=['World Bank-IDA', '13,054', 'China', '4,923'])]),
    dict(id='d06', split='dev', group='image_diagram',
         question='On the "How to Read the Aggregate and Country Profiles" page, which element of the annotated example profile does callout 3 mark, and what does that element measure?',
         answer='Callout 3 marks the total external debt stock: total debt outstanding and disbursed owed to nonresidents at end-2025, with the percent change from end-2024.',
         evidence=[dict(chunk='idr:p136:diagram', pdf_page=136, region='Annotated example profile, callout 3, and its legend text',
                        values=['Total external debt stock', 'owed to nonresidents at end-2025'])]),
    dict(id='d07', split='dev', group='chart',
         question='In Figure 1.1 (External Debt Stocks of Low- and Middle-Income Countries, 2015-25), which series peaks in 2021 and then declines through 2025?',
         answer="China's external debt stock (about US$2.7 trillion in 2021, falling to about US$2.3 trillion in 2025).",
         evidence=[dict(chunk='idr:p030:Figure_1.1', pdf_page=30, region='Figure 1.1, grey "China" line',
                        visual='China line highest at 2021 (~2.7 trn) then declines to ~2.3 trn in 2025; other lines keep rising')]),
    dict(id='d08', split='dev', group='chart',
         question="In Figure B1.4.1, which creditor type makes up the largest part of Ukraine's long-term external debt stock in 2025?",
         answer='Multilateral creditors (the largest band in 2025, larger than private creditors at about US$82 billion; total about US$227 billion).',
         mode='any',
         evidence=[dict(chunk='idr:p039:Figure_B1.4.1', pdf_page=39, region='Figure B1.4.1, 2025 stacked area',
                        visual='2025: private ~82, multilateral band ~82->213 (~130), bilateral thin top ~227'),
                   dict(pdf_page=38, quote='Debt owed to multilateral creditors accounts for most of the increase')]),
    dict(id='d09', split='dev', group='chart',
         question="In Kenya's country profile (Figure 3, Debt ratios), what was the external debt to exports ratio in 2025 and how did it change from 2024?",
         answer='212 percent, up 1.5 percent from 2024.',
         evidence=[dict(chunk='idr:p202:profile', pdf_page=202, region='Kenya profile, Figure 3, panel "External debt / exports"',
                        values=['212%', '+1.5%'])]),
    dict(id='d10', split='dev', group='multihop',
         question="How much larger was Azerbaijan's total external debt stock than Belize's at end-2025?",
         answer='US$10,718 million larger (Azerbaijan US$12,327 million vs Belize US$1,609 million).',
         calc='12,327 - 1,609 = 10,718', mode='all',
         evidence=[dict(chunk='idr:p150:profile', pdf_page=150, region='Azerbaijan profile header', values=['12,327']),
                   dict(chunk='idr:p153:profile', pdf_page=153, region='Belize profile header', values=['1,609'])]),
    dict(id='d11', split='dev', group='multihop',
         question='Does the 2024-to-2025 percentage change in LMIC external debt stock computed from Table 1.1 match the change reported on the "All Low- and Middle-Income Countries" profile page?',
         answer='Yes. Table 1.1: 8,949.6 -> 9,445.0 billion = +5.5 percent; the profile reports +5.5% from 2024.',
         calc='(9,445.0 - 8,949.6) / 8,949.6 = 5.54%', mode='all',
         evidence=[dict(chunk=T11, pdf_page=28, region='Table 1.1, row "Debt stock", 2024 and 2025', values=['8,949.6', '9,445.0']),
                   dict(chunk='idr:p137:profile', pdf_page=137, region='All LMICs profile header', values=['+5.5% from 2024'])]),
    dict(id='d12', split='dev', group='multihop',
         question='What was the external debt stock of IDA-eligible countries in 2025, and what were net debt inflows to IDA-eligible countries that year?',
         answer='About US$1.3 trillion (up 8.1 percent); net debt inflows were US$66.9 billion in 2025.',
         mode='all',
         # alt_chunks: đoạn Chương 1 (PDF p.62, p.34) nêu cùng số liệu — bổ sung sau khi đọc dev (thiếu sót của gold).
         evidence=[dict(pdf_page=19, quote='rising 8.1 percent to $1.3 trillion', alt_chunks=['idr:p062:text:0', 'idr:p034:text:2']),
                   dict(chunk=T12, pdf_page=63, region='Table 1.2, row "Net debt inflows", 2025', values=['66.9'])]),
    dict(id='d13', split='dev', group='text',
         question="What share of the increase in LMICs' external public and publicly guaranteed debt in 2025 was accounted for by the depreciation of the US dollar?",
         answer='28.2 percent.',
         evidence=[dict(pdf_page=19, quote='accounted for 28.2 percent of the increase in LMICs’ external public and publicly guaranteed debt')]),
    dict(id='d14', split='dev', group='text',
         question='In 2025, what share of new disbursements did LMIC public sector bond issuances account for, and what was their volume?',
         answer='68.1 percent of new disbursements; issuance rose 34.6 percent to US$250.6 billion.',
         evidence=[dict(pdf_page=20, quote='accounted for 68.1 percent of new disbursements in 2025, increasing 34.6 percent to $250.6 billion')]),
    dict(id='d15', split='dev', group='text',
         question="How much did LMICs' total debt service payments change in 2025, and how much was paid to bondholders?",
         answer='Total debt service rose 1.9 percent to US$1.4 trillion (fifth consecutive rise); payments to bondholders rose 13.7 percent to US$363.1 billion.',
         evidence=[dict(pdf_page=20, quote='increasing 1.9 percent to $1.4 trillion in 2025'),
                   dict(pdf_page=20, quote='payments to bondholders rising 13.7 percent to $363.1 billion')], mode='all'),
    dict(id='du1', split='dev', group='unanswerable',
         question="What does the report forecast for LMICs' total external debt stock in 2027?",
         absence="Report has no debt-stock forecasts: '2027' only appears as bond maturity years (PDF pp.43, 64, 72-73); "
                 "'projected' only for IDA interest costs without a figure (p.67). Data end at 2025 plus preliminary Jan-Aug 2026 bond data."),
    dict(id='du2', split='dev', group='unanswerable',
         question='What was the population of Kenya in 2015?',
         absence="Kenya population appears only on its profile as 'Population 57.5 million' for the current year (PDF p.202); no 2015 population anywhere."),
    dict(id='du3', split='dev', group='unanswerable',
         question='How much did Ukraine receive in grants from official creditors during 2022-25?',
         absence="Box 1.4 (PDF p.38) says support came 'through grants, trust funds, guarantees ...' and gives $122.2 billion in disbursements, "
                 "but no grant amount; regex 'grant ... $/billion/million' has 0 hits."),

    # ------------------------------- TEST (24 có đáp án + 6 thiếu bằng chứng) -------------------------------
    dict(id='t01', split='test', group='table',
         question='According to Table 1.1, what was LMICs\' external debt owed to the IMF (use of credit and SDR allocations) in 2021?',
         answer='US$386.4 billion.',
         evidence=[dict(chunk=T11, pdf_page=28, region='Table 1.1, row "IMF (Use of credit and SDR allocations)", column 2021', values=['386.4'])]),
    dict(id='t02', split='test', group='table',
         question='Using Table 1.1, by how much did long-term external debt owed to bondholders (Bonds) increase between 2015 and 2025?',
         answer='By US$1,122.2 billion, from US$1,242.2 billion to US$2,364.4 billion (roughly +90 percent).',
         calc='2,364.4 - 1,242.2 = 1,122.2; 1,122.2 / 1,242.2 = 90.3%',
         evidence=[dict(chunk=T11, pdf_page=28, region='Table 1.1, row "Bonds", columns 2015 and 2025', values=['1,242.2', '2,364.4'])]),
    dict(id='t03', split='test', group='table',
         question='According to Table 1.2, in which year were short-term net debt inflows to IDA-eligible countries most negative, and what was the value?',
         answer='2023, at -US$11.0 billion.',
         evidence=[dict(chunk=T12, pdf_page=63, region='Table 1.2, row "Short-term", 2015-2025 (min = 2023)', values=['11.0'])]),
    dict(id='t04', split='test', group='table',
         question='According to Table 1.2, in which year did net inflows from the World Bank to IDA-eligible countries peak during 2015-25, and at what level?',
         answer='2024, at US$18.3 billion.',
         evidence=[dict(chunk=T12, pdf_page=63, region='Table 1.2, row "World Bank", 2015-2025 (max = 2024)', values=['18.3'])]),
    dict(id='t05', split='test', group='table',
         question="Using Table 1.1, what share of LMICs' total external debt stock in 2025 was owed to multilateral creditors?",
         answer='About 15.8 percent (US$1,493.7 billion of US$9,445.0 billion).',
         calc='1,493.7 / 9,445.0 = 15.8%',
         evidence=[dict(chunk=T11, pdf_page=28, region='Table 1.1, rows "Multilateral creditors" and "Debt stock", column 2025', values=['1,493.7', '9,445.0'])]),
    dict(id='t06', split='test', group='table',
         question='According to Table 1.2, in 2025 were net debt inflows to IDA-eligible countries from private creditors larger or smaller than those from official creditors, and by how much?',
         answer='Smaller: private creditors US$23.9 billion vs official creditors US$30.3 billion, a gap of US$6.4 billion.',
         calc='30.3 - 23.9 = 6.4',
         evidence=[dict(chunk=T12, pdf_page=63, region='Table 1.2, rows "Official creditors" and "Private creditors", 2025', values=['30.3', '23.9'])]),
    dict(id='t07', split='test', group='image_diagram',
         question='In the three-tier architecture diagram of the World Bank Group Debt Statistics Program, what is Tier 3 and how is it described?',
         answer='Tier 3 is the IDR (International Debt Report), described as "Actionable knowledge".',
         evidence=[dict(chunk='idr:p132:Figure_3B.1', pdf_page=132, region='Figure 3B.1, right (orange) box', values=['Tier 3', 'IDR', 'Actionable knowledge'])]),
    dict(id='t08', split='test', group='image_diagram',
         question='In the "All Low- and Middle-Income Countries" profile creditor diagram (Figure 2), which named bilateral creditor has the largest claims and how much?',
         answer='China, US$144 billion (Japan is US$112 billion; other bilateral US$244 billion is a residual group).',
         evidence=[dict(chunk='idr:p137:profile', pdf_page=137, region='All LMICs profile, Figure 2, BILATERAL row (China 144 | Japan 112 | Other bilateral 244)',
                        values=['144', '112'], visual='Text layer order is "Japan | 144 | 112"; the image shows the 144 block labelled China and 112 labelled Japan')]),
    dict(id='t09', split='test', group='image_diagram',
         question="In Viet Nam's profile creditor diagram (Figure 2), how much public and publicly guaranteed debt is owed to private creditors, and which bilateral creditor other than Japan is named?",
         answer='US$1,263 million is owed to private creditors; the Russian Federation is named, with US$1,392 million.',
         evidence=[dict(chunk='idr:p260:profile', pdf_page=260, region='Viet Nam profile, Figure 2, legend items under PPG-by-creditor and BILATERAL rows',
                        values=['Private creditors 1,263', 'Russian Federation 1,392'])]),
    dict(id='t10', split='test', group='image_diagram',
         question='Which country is used as the annotated example on the "How to Read the Aggregate and Country Profiles" page, and what total external debt stock and change from 2024 does the example show?',
         answer='Belize: total external debt stock of 1,609 (US$ million), +2.6% from 2024.',
         # Chỉ trang hướng dẫn cho biết nước nào là ví dụ; hồ sơ Belize (PDF p.153) chỉ xác nhận cùng số liệu.
         evidence=[dict(chunk='idr:p136:diagram', pdf_page=136, region='Annotated example image (top right) with callouts 1-7',
                        values=['BELIZE', '1,609', '+2.6% from 2024'])]),
    dict(id='t11', split='test', group='image_diagram',
         question="In Sri Lanka's profile creditor diagram (Figure 2), which is the largest bilateral creditor, and are its claims larger than those of the Asian Development Bank?",
         answer='China, US$8,766 million — larger than the Asian Development Bank (US$7,379 million).',
         evidence=[dict(chunk='idr:p241:profile', pdf_page=241, region='Sri Lanka profile, Figure 2, MULTILATERAL and BILATERAL rows',
                        values=['China', '8,766', 'Asian Dev. Bank', '7,379'])]),
    dict(id='t12', split='test', group='chart',
         question='In Figure 3.2 (Debtor Reporting System Reporting Timetable, 2020-25), in which year was the share of countries reporting by March highest, and was the share of late reporters and nonreporters higher in 2020 or in 2025?',
         answer='March reporting was highest in 2025 (about 56 percent); late reporters/nonreporters were higher in 2020 (about 13 percent vs about 6 percent in 2025).',
         evidence=[dict(chunk='idr:p111:Figure_3.2', pdf_page=111, region='Figure 3.2, dark-blue "March" bars and gold "Late reporters and nonreporters" bars',
                        visual='March: 2020~47, 2021~45, 2022~48, 2023~51, 2024~48, 2025~56; Late: 2020~13 ... 2025~6')]),
    dict(id='t13', split='test', group='chart',
         question='In Figure 1.6 (Composition of Debt Flows, Top 10 Borrowers Excluding China), which country had the largest positive debt flows and which component accounted for most of them?',
         answer='Türkiye; private nonguaranteed flows (about US$45 billion of a bar reaching about US$69 billion, partly offset by negative short-term flows).',
         evidence=[dict(chunk='idr:p046:Figure_1.6', pdf_page=46, region='Figure 1.6, Türkiye stacked bar',
                        visual='Türkiye bar tallest (~69); light-blue PNG segment ~24->69; orange short-term below zero (~-6)')]),
    dict(id='t14', split='test', group='chart',
         question='According to Figure 2.10, how many LIC-DSF external risk rating downgrades and upgrades were there in total over 2020-25, and in which single year were upgrades highest?',
         answer='22 downgrades and 15 upgrades in total; upgrades were highest in 2025 (6).',
         evidence=[dict(chunk='idr:p088:Figure_2.10', pdf_page=88, region='Figure 2.10, panels a and b, final 2020-25 bars and yearly labels',
                        values=['22', '15', '6'])]),
    dict(id='t15', split='test', group='chart',
         question='In Figure 1.15 (Remittances, by Region, 2019-25), which region received the most remittances in 2025, and did remittances to Sub-Saharan Africa rise or fall between 2024 and 2025?',
         answer='South Asia (about US$207 billion); remittances to Sub-Saharan Africa fell (from about US$63 billion to about US$52 billion).',
         evidence=[dict(chunk='idr:p058:Figure_1.15', pdf_page=58, region='Figure 1.15, 2024 and 2025 bar groups',
                        visual='2025: South Asia (grey) tallest ~207; SSA (dark navy) 2024 ~63 -> 2025 ~52')]),
    dict(id='t16', split='test', group='chart',
         question='According to Figure 3.3 (Three Years of the Creditor Data-Sharing Exercise), how many loans and tranches had been reported by end-2024, and how many of them were matched?',
         answer='4,875 loans and tranches reported, of which 3,762 were matched.',
         evidence=[dict(chunk='idr:p113:Figure_3.3', pdf_page=113, region='Figure 3.3, End-2024 line markers (right axis)', values=['4,875', '3,762'])]),
    dict(id='t17', split='test', group='multihop',
         question="Which had the larger total external debt stock at end-2025, Kenya or Sri Lanka, and by how much?",
         answer='Sri Lanka, by US$11,345 million (US$57,658 million vs US$46,313 million).',
         calc='57,658 - 46,313 = 11,345', mode='all',
         evidence=[dict(chunk='idr:p202:profile', pdf_page=202, region='Kenya profile header', values=['46,313']),
                   dict(chunk='idr:p241:profile', pdf_page=241, region='Sri Lanka profile header', values=['57,658'])]),
    dict(id='t18', split='test', group='multihop',
         question="In 2020, what were net debt inflows from the IMF to IDA-eligible countries, and what was all LMICs' outstanding external debt owed to the IMF (use of credit and SDR allocations)?",
         answer='Net IMF inflows to IDA-eligible countries: US$15.2 billion (Table 1.2); LMIC debt owed to the IMF: US$214.6 billion (Table 1.1).',
         mode='all',
         evidence=[dict(chunk=T12, pdf_page=63, region='Table 1.2, row "IMF", column 2020', values=['15.2']),
                   dict(chunk=T11, pdf_page=28, region='Table 1.1, row "IMF (Use of credit and SDR allocations)", column 2020', values=['214.6'])]),
    dict(id='t19', split='test', group='multihop',
         question="What share of Ukraine's total external debt stock at end-2025 was long-term external debt?",
         answer='About 90.7 percent (long-term US$227 billion of a total US$250.3 billion).',
         calc='227 / 250.3 = 90.7%', mode='all',
         evidence=[dict(pdf_page=31, quote='increase in Ukraine’s debt stock, to $250.3 billion'),
                   dict(pdf_page=38, quote='Long-term external debt stock rose from $99 billion in 2015 to $227 billion in 2025')]),
    dict(id='t20', split='test', group='multihop',
         question="Viet Nam's total external debt stock in 2025 was what share of the East Asia and Pacific regional total?",
         answer='About 4.0 percent (US$131,777 million = US$131.8 billion out of US$3,313 billion).',
         calc='131,777 million = 131.777 billion; 131.777 / 3,313 = 3.98%', mode='all',
         evidence=[dict(chunk='idr:p260:profile', pdf_page=260, region='Viet Nam profile header (US$ million)', values=['131,777']),
                   dict(chunk='idr:p138:profile', pdf_page=138, region='East Asia and Pacific profile header (US$ billion)', values=['3,313'])]),
    dict(id='t21', split='test', group='text',
         question='Since 2024, in which countries have total return swap (TRS) arrangements been identified, and within what range have disclosed collateralization ratios typically fallen?',
         answer='Angola, Argentina, Colombia, Nigeria, and Senegal; typically between 133 and 198 percent.',
         evidence=[dict(pdf_page=32, quote='TRS arrangements have been identified in Angola, Argentina, Colombia, Nigeria, and Senegal'),
                   dict(pdf_page=32, quote='collateralization ratios have typically been between 133 and 198 percent')], mode='all'),
    dict(id='t22', split='test', group='text',
         question='What average interest rate did IDA-eligible countries face on new commitments in 2025, and how did the gap between their official and private borrowing rates change since 2016?',
         answer='8.2 percent (highest since the late 1980s); the gap widened from 3.4 percentage points in 2016 to 5.5 percentage points in 2025.',
         evidence=[dict(pdf_page=20, quote='faced an average interest rate of 8.2 percent on new commitments'),
                   dict(pdf_page=20, quote='from 3.4 percentage points in 2016 to 5.5 percentage points in 2025')], mode='all'),
    dict(id='t23', split='test', group='text',
         question="Under the G-7 Extraordinary Revenue Acceleration (ERA) initiative, how are the loans to Ukraine serviced, and why are they still recorded as Ukraine's external debt?",
         answer='Principal and interest are serviced from future revenues generated by immobilized Russian sovereign assets (channeled through the Ukraine Loan Cooperation Mechanism); they are still debt because classification depends on the contractual obligation to repay, not on the expected source of repayment.',
         evidence=[dict(pdf_page=39, quote='serviced from future revenues generated by immobilized Russian sovereign assets'),
                   dict(pdf_page=39, quote='based on the existence of a contractual obligation to repay principal and/or interest')], mode='all'),
    dict(id='t24', split='test', group='text',
         question="What were the terms of Kenya's October 2025 dual-tranche Eurobond issuance?",
         answer='US$1.5 billion in two equal US$750 million tranches: a 7.875 percent note due 2033 and an 8.80 percent note due 2038.',
         evidence=[dict(pdf_page=73, quote='comprising two equal $750 million tranches')]),
    dict(id='tu1', split='test', group='unanswerable',
         question="What was Viet Nam's total external debt stock at end-2026?",
         absence="Profiles cover 2015-25 only (Viet Nam PDF p.260); regex 'Viet Nam ... 2026' has 0 hits in text layer and vision descriptions."),
    dict(id='tu2', split='test', group='unanswerable',
         question="What was Germany's total external debt stock in 2025?",
         absence="Report covers LMICs; Germany appears only as a bilateral creditor (e.g. Albania, Argentina, Armenia profiles, PDF p.70 Paris Club text), never with its own debt stock."),
    dict(id='tu3', split='test', group='unanswerable',
         question='What interest rate does Ukraine pay on the G-7 Extraordinary Revenue Acceleration (ERA) loans?',
         absence="Box 1.4 (PDF pp.38-40) describes the ERA (~$50 billion, serviced from immobilized Russian asset revenues) but states no interest rate; "
                 "regex 'ERA ... interest rate' has 0 relevant hits."),
    dict(id='tu4', split='test', group='unanswerable',
         question="What was Kenya's total external debt stock in 2010?",
         absence="Kenya profile Figure 1 covers 2015-25 (2015 = 19,788; 2025 = 46,313); regex 'Kenya ... 2010' has 0 hits."),
    dict(id='tu5', split='test', group='unanswerable',
         question='How much external debt did India owe to China at end-2025?',
         absence="India profile (PDF p.195) names only Japan (27,445) and the Russian Federation (5,565) as bilateral creditors; China is folded into "
                 "'Other bilateral 6,115' with no separate figure."),
    dict(id='tu6', split='test', group='unanswerable',
         question="What was Belize's real GDP growth rate in 2025?",
         absence="Belize profile (PDF p.153) gives GNI 3,162 and debt ratios but no GDP growth; 'GDP growth' appears once in the report (p.87, generic)."),
]


def norm(s: str) -> str:
    s = s.replace('\u00a0', ' ').replace('‘', "'").replace('’', "'").replace('–', '-').replace('−', '-')
    s = re.sub(r'-\s*\n\s*', '-', s)  # gạch nối cuối dòng
    return re.sub(r'\s+', ' ', s).strip().lower()


def main() -> None:
    doc = pymupdf.open(PDF)
    chunks = json.loads((OUT / 'chunks.json').read_text(encoding='utf-8'))
    by_id = {c['id']: c for c in chunks}
    problems = []
    for q in Q:
        if q['group'] == 'unanswerable':
            q.update(answerable=False, expected_behavior='search_and_abstain')
            continue
        q.update(answerable=True, expected_behavior='search_and_answer')
        q.setdefault('mode', 'all' if len(q['evidence']) > 1 else 'any')
        for e in q['evidence']:
            page_text = doc[e['pdf_page'] - 1].get_text()
            e['printed_page'] = doc[e['pdf_page'] - 1].get_label()
            layer = norm(page_text)
            for v in e.get('values', []):
                if norm(v) not in layer:
                    problems.append(f"{q['id']}: '{v}' không có trong text layer gốc PDF p.{e['pdf_page']}")
            if 'quote' in e:
                if norm(e['quote']) not in layer:
                    problems.append(f"{q['id']}: quote không có trong text layer gốc PDF p.{e['pdf_page']}")
                hits = [c['id'] for c in chunks if c['page'] == e['pdf_page'] and norm(e['quote']) in norm(c['text'])]
                if not hits:
                    problems.append(f"{q['id']}: quote không nằm trọn trong chunk nào của p.{e['pdf_page']}")
                else:
                    e['chunk'] = hits[0]
            for alt in e.get('alt_chunks', []):
                if alt not in by_id:
                    problems.append(f"{q['id']}: alt chunk {alt} không tồn tại")
            if e.get('chunk') not in by_id:
                problems.append(f"{q['id']}: chunk {e.get('chunk')} không tồn tại")
    if problems:
        raise SystemExit('Benchmark lỗi:\n' + '\n'.join(problems))
    from collections import Counter
    out = OUT / 'benchmark.json'
    out.write_text(json.dumps(Q, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    for split in ('dev', 'test'):
        print(split, dict(Counter(q['group'] for q in Q if q['split'] == split)), len([q for q in Q if q['split'] == split]))
    ids = [q['id'] for q in Q]
    assert len(ids) == len(set(ids))
    print('OK ->', out.relative_to(config.ROOT), '| mọi values/quote khớp text layer gốc và chunk tồn tại')


if __name__ == '__main__':
    main()
