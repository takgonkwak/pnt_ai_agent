"""
MES DB SQL 쿼리문 관리 모듈
"""

# Lot 정보 조회
QUERY_LOT = """
SELECT 
    LOT_ID, 
    PRODUCT_ID, 
    PROCESS_ID, 
    EQUIP_ID, 
    STATUS, 
    QTY, 
    HOLD_REASON, 
    LAST_UPDATED 
FROM LOT_MASTER 
WHERE LOT_ID = :lot_id
"""

# 설비 상태 조회
QUERY_EQUIPMENT = """
SELECT 
    EQUIP_ID, 
    EQUIP_NAME, 
    STATUS, 
    CURRENT_RECIPE, 
    ALARM_CODE, 
    ALARM_MESSAGE, 
    LAST_PM 
FROM EQUIP_STATUS 
WHERE EQUIP_ID = :equip_id
"""

# WIP(재공) 현황 조회 베이스
# WHERE 절은 서버 단에서 파라미터 유무에 따라 동적으로 구성됨
QUERY_WIP_BASE = """
SELECT 
    LOT_ID, 
    PRODUCT_ID, 
    PROCESS_ID, 
    EQUIP_ID, 
    STATUS, 
    QTY 
FROM WIP_STATUS
"""
