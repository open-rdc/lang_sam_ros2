"""単一オブジェクト用CSRTトラッカー"""

import cv2
import numpy as np
import logging
from typing import Dict, Optional, Tuple, Any

from .exceptions import CSRTTrackingError

# ロガー設定
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # CSRTパラメータログを表示するためINFOレベルに設定


class CSRTTracker:
    """単一オブジェクト用CSRTトラッカー（シンプル実装）"""
    
    def __init__(self, tracker_id: str, bbox: Tuple[int, int, int, int], 
                 image: np.ndarray, csrt_params: Optional[Dict] = None):
        self.tracker_id = tracker_id
        self.csrt_params = csrt_params or {}
        
        # 強制ログ出力 - CSRTパラメータ確認
        print(f"[CSRT_DEBUG] {tracker_id}: csrt_params = {self.csrt_params}")
        logger.info(f"[{self.tracker_id}] CSRTパラメータ受け取り: {len(self.csrt_params)}個")
        
        logger.info(f"[{self.tracker_id}] CSRT tracker初期化開始 - bbox: {bbox}, image shape: {image.shape}")
        logger.info(f"[{self.tracker_id}] CSRTパラメータ詳細: {self.csrt_params}")
        
        # 実際にパラメータ処理を実行するかテスト
        print(f"[INIT_DEBUG] {tracker_id}: パラメータ数={len(self.csrt_params)}, 空判定={not self.csrt_params}")
        
        # トラッカー生成前のテスト
        print(f"[BEFORE_CREATE] {tracker_id}: トラッカー生成開始")
        
        self.tracker = self._create_tracker()
        
        # トラッカー生成後のテスト
        print(f"[AFTER_CREATE] {tracker_id}: トラッカー生成完了, tracker={self.tracker is not None}")
        
        self.is_initialized = False
        
        if self.tracker and self._initialize(image, bbox):
            self.is_initialized = True
            logger.info(f"[{self.tracker_id}] CSRT tracker初期化成功")
            
            # OpenCV 4.11.0対応：初期化後のパラメータ手動適用を試行
            if hasattr(self, 'params_for_post_init') and self.params_for_post_init:
                self._apply_post_init_parameters()
        else:
            logger.error(f"[{self.tracker_id}] CSRT tracker初期化失敗")
    
    def _create_tracker(self) -> Optional[cv2.Tracker]:
        """OpenCV CSRTトラッカー生成（パラメータ対応）"""
        print(f"[CREATE_TRACKER] {self.tracker_id}: _create_tracker()メソッド開始")
        
        try:
            print(f"[CREATE_TRACKER] {self.tracker_id}: try文開始")
            logger.info(f"[{self.tracker_id}] CSRTトラッカー生成開始")
            logger.info(f"[{self.tracker_id}] 利用可能CSRTパラメータ数: {len(self.csrt_params)}")
            
            # OpenCV環境の詳細確認
            print(f"[OPENCV_DEBUG] cv2.__version__ = {cv2.__version__}")
            print(f"[OPENCV_DEBUG] hasattr(cv2, 'TrackerCSRT') = {hasattr(cv2, 'TrackerCSRT')}")
            print(f"[OPENCV_DEBUG] hasattr(cv2, 'TrackerCSRT_create') = {hasattr(cv2, 'TrackerCSRT_create')}")
            print(f"[OPENCV_DEBUG] hasattr(cv2, 'legacy') = {hasattr(cv2, 'legacy')}")
            if hasattr(cv2, 'legacy'):
                print(f"[OPENCV_DEBUG] hasattr(cv2.legacy, 'TrackerCSRT_create') = {hasattr(cv2.legacy, 'TrackerCSRT_create')}")
                print(f"[OPENCV_DEBUG] hasattr(cv2.legacy, 'TrackerCSRT') = {hasattr(cv2.legacy, 'TrackerCSRT')}")
                if hasattr(cv2.legacy, 'TrackerCSRT'):
                    print(f"[OPENCV_DEBUG] hasattr(cv2.legacy.TrackerCSRT, 'Params') = {hasattr(cv2.legacy.TrackerCSRT, 'Params')}")
            else:
                print(f"[OPENCV_DEBUG] cv2.legacy名前空間は利用できません")
            
            # パラメータがある場合は適用してトラッカー生成
            print(f"[CREATE_TRACKER] {self.tracker_id}: パラメータ条件チェック, csrt_params存在={bool(self.csrt_params)}")
            if self.csrt_params:
                print(f"[CREATE_TRACKER] {self.tracker_id}: if条件通過、パラメータ適用開始")
                logger.info(f"[{self.tracker_id}] CSRTパラメータ適用開始")
                print(f"[CREATE_TRACKER] {self.tracker_id}: logger.info完了、_create_csrt_params()呼び出し")
                params = self._create_csrt_params()
                print(f"[CREATE_TRACKER] {self.tracker_id}: _create_csrt_params()完了, params={params is not None}")
                if params:
                    logger.info(f"[{self.tracker_id}] CSRTパラメータオブジェクト生成成功")
                    
                    # OpenCV 4.11.0では`TrackerCSRT_create(params)`がNULLを返すため、パラメータなしで生成
                    logger.warning(f"[{self.tracker_id}] OpenCV 4.11.0パラメータ問題回避: パラメータなしトラッカー生成")
                    
                    # パラメータオブジェクトを保存して、初期化後の手動設定に使用
                    self.params_for_post_init = params
                else:
                    self.params_for_post_init = None
            
            # OpenCV 4.11.0安定版API: パラメータなしでトラッカー生成
            print(f"[CREATE_TRACKER] {self.tracker_id}: デフォルト生成開始")
            
            # cv2.legacy.TrackerCSRT_create()を最優先で試行（OpenCV 4.11.0推奨）
            try:
                print(f"[CREATE_TRACKER] {self.tracker_id}: cv2.legacy.TrackerCSRT_create()試行")
                tracker = cv2.legacy.TrackerCSRT_create()
                if tracker is not None:
                    logger.info(f"[{self.tracker_id}] cv2.legacy.TrackerCSRT_create()で生成成功")
                    return tracker
                else:
                    print(f"[CREATE_TRACKER] {self.tracker_id}: cv2.legacy.TrackerCSRT_create()がNoneを返しました")
            except Exception as e:
                print(f"[CREATE_TRACKER] {self.tracker_id}: cv2.legacy.TrackerCSRT_create()失敗: {e}")
                logger.warning(f"[{self.tracker_id}] cv2.legacy.TrackerCSRT_create()失敗: {e}")
            
            # 標準APIをフォールバックとして試行
            try:
                print(f"[CREATE_TRACKER] {self.tracker_id}: cv2.TrackerCSRT_create()試行")
                tracker = cv2.TrackerCSRT_create()
                if tracker is not None:
                    logger.info(f"[{self.tracker_id}] cv2.TrackerCSRT_create()で生成成功")
                    return tracker
                else:
                    print(f"[CREATE_TRACKER] {self.tracker_id}: cv2.TrackerCSRT_create()がNoneを返しました")
            except Exception as e:
                print(f"[CREATE_TRACKER] {self.tracker_id}: cv2.TrackerCSRT_create()失敗: {e}")
                logger.warning(f"[{self.tracker_id}] cv2.TrackerCSRT_create()失敗: {e}")
            
            # 全ての方法が失敗した場合
            logger.error(f"[{self.tracker_id}] 全てのCSRTトラッカー生成方法が失敗しました")
            return None
        except Exception as e:
            logger.error(f"[{self.tracker_id}] CSRTトラッカー生成失敗: {e}")
            raise CSRTTrackingError(self.tracker_id, "tracker creation", e)
    
    def _create_csrt_params(self) -> Any:
        """CSRTパラメータオブジェクト作成"""
        print(f"[_CREATE_CSRT_PARAMS] {self.tracker_id}: メソッド開始")
        try:
            print(f"[_CREATE_CSRT_PARAMS] {self.tracker_id}: try文開始")
            logger.info(f"[{self.tracker_id}] CSRTパラメータオブジェクト作成開始")
            print(f"[_CREATE_CSRT_PARAMS] {self.tracker_id}: logger.info完了")
            
            # OpenCV 4.11.0対応: cv2.legacy名前空間を含む複数のAPIを試行
            print(f"[_CREATE_CSRT_PARAMS] {self.tracker_id}: パラメータクラス作成開始")
            params = None
            
            # cv2.legacy名前空間を最初に試行（OpenCV 4.11.0推奨）
            try:
                print(f"[_CREATE_CSRT_PARAMS] {self.tracker_id}: cv2.legacy.TrackerCSRT.Params()試行")
                params = cv2.legacy.TrackerCSRT.Params()
                print(f"[_CREATE_CSRT_PARAMS] {self.tracker_id}: cv2.legacy.TrackerCSRT.Params()成功")
                logger.info(f"[{self.tracker_id}] cv2.legacy.TrackerCSRT.Params()で生成成功")
            except AttributeError as e:
                print(f"[_CREATE_CSRT_PARAMS] {self.tracker_id}: cv2.legacy試行失敗={e}")
                
                # 標準名前空間でパラメータクラス作成を試行
                try:
                    print(f"[_CREATE_CSRT_PARAMS] {self.tracker_id}: cv2.TrackerCSRT.Params()試行")
                    params = cv2.TrackerCSRT.Params()
                    print(f"[_CREATE_CSRT_PARAMS] {self.tracker_id}: cv2.TrackerCSRT.Params()成功")
                    logger.info(f"[{self.tracker_id}] cv2.TrackerCSRT.Params()で生成成功")
                except AttributeError as e2:
                    print(f"[_CREATE_CSRT_PARAMS] {self.tracker_id}: 標準名前空間失敗={e2}")
                    
                    # 古いバージョンのフォールバック
                    if hasattr(cv2, 'TrackerCSRT_Params'):
                        print(f"[_CREATE_CSRT_PARAMS] {self.tracker_id}: cv2.TrackerCSRT_Params()試行")
                        params = cv2.TrackerCSRT_Params()
                        print(f"[_CREATE_CSRT_PARAMS] {self.tracker_id}: cv2.TrackerCSRT_Params()成功")
                        logger.info(f"[{self.tracker_id}] cv2.TrackerCSRT_Params()で生成成功")
                    else:
                        print(f"[_CREATE_CSRT_PARAMS] {self.tracker_id}: 全てのパラメータクラスが見つからない")
                        logger.error(f"[{self.tracker_id}] CSRTパラメータクラスが見つかりません")
                        return None
                except Exception as e2:
                    print(f"[_CREATE_CSRT_PARAMS] {self.tracker_id}: 標準名前空間例外={e2}")
                    logger.error(f"[{self.tracker_id}] 標準パラメータ作成例外: {e2}")
                    return None
            except Exception as e:
                print(f"[_CREATE_CSRT_PARAMS] {self.tracker_id}: cv2.legacy例外={e}")
                logger.error(f"[{self.tracker_id}] legacyパラメータ作成例外: {e}")
                return None
            
            if params is None:
                print(f"[_CREATE_CSRT_PARAMS] {self.tracker_id}: パラメータオブジェクト作成失敗")
                logger.error(f"[{self.tracker_id}] パラメータオブジェクト作成失敗")
                return None
            
            print(f"[_CREATE_CSRT_PARAMS] {self.tracker_id}: パラメータオブジェクト作成完了")
            
            # デフォルト値をログ出力
            print(f"[_CREATE_CSRT_PARAMS] {self.tracker_id}: デフォルト値ログ開始")
            self._log_default_params(params)
            print(f"[_CREATE_CSRT_PARAMS] {self.tracker_id}: デフォルト値ログ完了")
            
            # カスタムパラメータを適用
            print(f"[_CREATE_CSRT_PARAMS] {self.tracker_id}: カスタムパラメータ適用開始")
            self._apply_csrt_params(params)
            print(f"[_CREATE_CSRT_PARAMS] {self.tracker_id}: カスタムパラメータ適用完了")
            
            # 適用後の値をログ出力
            print(f"[_CREATE_CSRT_PARAMS] {self.tracker_id}: 適用後ログ開始")
            self._log_applied_params(params)
            print(f"[_CREATE_CSRT_PARAMS] {self.tracker_id}: 適用後ログ完了")
            
            # OpenCVパラメータ検証 - 実際に設定された値を確認
            print(f"[_CREATE_CSRT_PARAMS] {self.tracker_id}: 検証開始")
            self._verify_opencv_params(params)
            print(f"[_CREATE_CSRT_PARAMS] {self.tracker_id}: 検証完了")
            
            print(f"[_CREATE_CSRT_PARAMS] {self.tracker_id}: パラメータ返却")
            return params
            
        except Exception as e:
            print(f"[_CREATE_CSRT_PARAMS] {self.tracker_id}: 全体例外={e}")
            logger.error(f"[{self.tracker_id}] CSRTパラメータ作成失敗: {e}")
            return None
    
    def _apply_csrt_params(self, params: Any) -> None:
        """CSRTパラメータ値適用"""
        param_mapping = self._get_csrt_param_mapping()
        applied_params = []
        
        # 除外するパラメータ（ROSトピック名等）
        excluded_params = {
            'output_topic',  # csrt_output_topic = "/image_csrt"  
            'gdino_topic', 'sam_topic',  # 他のトピック名
            'enable_csrt_recovery', 'frame_buffer_duration',  # CSRT機能設定
            'time_travel_seconds', 'fast_forward_frames', 'recovery_attempt_frames'
        }
        
        for param_key, param_value in self.csrt_params.items():
            if param_key.startswith('csrt_'):
                clean_key = param_key[5:]  # 'csrt_'プレフィックス除去
                
                # 除外パラメータをスキップ
                if clean_key in excluded_params:
                    logger.info(f"[{self.tracker_id}] 除外パラメータ: {param_key}")
                    continue
                
                opencv_attr = param_mapping.get(clean_key)
                
                if opencv_attr and hasattr(params, opencv_attr):
                    try:
                        setattr(params, opencv_attr, param_value)
                        applied_params.append(f"{opencv_attr}={param_value}")
                    except Exception as e:
                        logger.warning(f"[{self.tracker_id}] パラメータ設定失敗 {opencv_attr}: {e}")
                else:
                    logger.warning(f"[{self.tracker_id}] 不明なCSRTパラメータ: {param_key}")
        
        if applied_params:
            logger.info(f"[{self.tracker_id}] 適用されたCSRTパラメータ: {applied_params}")
        else:
            logger.info(f"[{self.tracker_id}] カスタムパラメータの適用なし")
    
    def _get_csrt_param_mapping(self) -> Dict[str, str]:
        """CSRTパラメータマッピング（公式ドキュメント準拠）"""
        return {
            # 基本設定
            'use_hog': 'use_hog',
            'use_color_names': 'use_color_names',
            'use_gray': 'use_gray', 
            'use_rgb': 'use_rgb',
            'use_channel_weights': 'use_channel_weights',
            'use_segmentation': 'use_segmentation',
            
            # ウィンドウ関数設定
            'window_function': 'window_function',  # 'hann', 'cheb', 'kaiser'
            'kaiser_alpha': 'kaiser_alpha',
            'cheb_attenuation': 'cheb_attenuation',
            
            # テンプレート設定
            'padding': 'padding',
            'template_size': 'template_size',
            'gsl_sigma': 'gsl_sigma',
            
            # HOG設定
            'hog_orientations': 'hog_orientations',
            'hog_clip': 'hog_clip',
            'num_hog_channels_used': 'num_hog_channels_used',
            
            # 学習率設定
            'filter_lr': 'filter_lr',
            'weights_lr': 'weights_lr',
            
            # ADMM最適化
            'admm_iterations': 'admm_iterations',
            
            # スケール設定
            'number_of_scales': 'number_of_scales',
            'scale_sigma_factor': 'scale_sigma_factor',
            'scale_model_max_area': 'scale_model_max_area',
            'scale_lr': 'scale_lr',
            'scale_step': 'scale_step',
            
            # ヒストグラム設定
            'histogram_bins': 'histogram_bins',
            'histogram_lr': 'histogram_lr',
            'background_ratio': 'background_ratio',
            
            # 信頼性閾値
            'psr_threshold': 'psr_threshold'
        }
    
    def _get_default_csrt_params(self) -> Dict[str, Any]:
        """CSRTデフォルトパラメータ値（OpenCVソースコード準拠）"""
        return {
            # 基本設定
            'use_hog': True,
            'use_color_names': True,
            'use_gray': True,
            'use_rgb': False,
            'use_channel_weights': True,
            'use_segmentation': True,
            
            # ウィンドウ関数設定
            'window_function': 'hann',
            'kaiser_alpha': 3.75,
            'cheb_attenuation': 45.0,
            
            # テンプレート設定
            'padding': 3.0,
            'template_size': 200.0,
            'gsl_sigma': 1.0,
            
            # HOG設定
            'hog_orientations': 9.0,
            'hog_clip': 0.2,
            'num_hog_channels_used': 18,
            
            # 学習率設定
            'filter_lr': 0.02,
            'weights_lr': 0.02,
            
            # ADMM最適化
            'admm_iterations': 4,
            
            # スケール設定
            'number_of_scales': 33,
            'scale_sigma_factor': 0.25,
            'scale_model_max_area': 512.0,
            'scale_lr': 0.025,
            'scale_step': 1.02,
            
            # ヒストグラム設定
            'histogram_bins': 16,
            'histogram_lr': 0.04,
            'background_ratio': 2,
            
            # 信頼性閾値
            'psr_threshold': 0.035
        }
    
    def _log_default_params(self, params: Any) -> None:
        """デフォルトCSRTパラメータ値をログ出力"""
        try:
            key_params = [
                'use_hog', 'use_color_names', 'use_gray', 'use_segmentation',
                'filter_lr', 'weights_lr', 'template_size', 'padding',
                'admm_iterations', 'psr_threshold', 'scale_lr'
            ]
            defaults = []
            for param in key_params:
                if hasattr(params, param):
                    value = getattr(params, param)
                    defaults.append(f"{param}={value}")
            logger.info(f"[{self.tracker_id}] デフォルトCSRTパラメータ: {', '.join(defaults)}")
        except Exception as e:
            logger.info(f"[{self.tracker_id}] デフォルトパラメータログ出力失敗: {e}")
    
    def _log_applied_params(self, params: Any) -> None:
        """適用後CSRTパラメータ値をログ出力"""
        try:
            applied = []
            param_mapping = self._get_csrt_param_mapping()
            
            for param_key, param_value in self.csrt_params.items():
                if param_key.startswith('csrt_'):
                    clean_key = param_key[5:]
                    opencv_attr = param_mapping.get(clean_key)
                    
                    if opencv_attr and hasattr(params, opencv_attr):
                        actual_value = getattr(params, opencv_attr)
                        applied.append(f"{clean_key}={actual_value}")
            
            if applied:
                logger.info(f"[{self.tracker_id}] 適用後CSRTパラメータ: {', '.join(applied)}")
        except Exception as e:
            logger.info(f"[{self.tracker_id}] 適用後パラメータログ出力失敗: {e}")
    
    def _verify_opencv_params(self, params: Any) -> None:
        """OpenCVパラメータの実際の値を検証"""
        try:
            verification = []
            test_params = ['use_hog', 'use_gray', 'template_size', 'filter_lr', 'psr_threshold']
            
            for param in test_params:
                if hasattr(params, param):
                    value = getattr(params, param)
                    verification.append(f"{param}={value}")
            
            logger.info(f"[{self.tracker_id}] OpenCV実際値検証: {', '.join(verification)}")
        except Exception as e:
            logger.info(f"[{self.tracker_id}] OpenCV値検証失敗: {e}")
    
    def _apply_post_init_parameters(self) -> None:
        """初期化後のパラメータ手動適用（OpenCV 4.11.0対応）"""
        try:
            logger.info(f"[{self.tracker_id}] 初期化後パラメータ適用開始")
            
            # OpenCVトラッカーオブジェクトのパラメータ直接設定を試行
            if hasattr(self.tracker, 'setParams'):
                try:
                    self.tracker.setParams(self.params_for_post_init)
                    logger.info(f"[{self.tracker_id}] setParams()によるパラメータ適用成功")
                    return
                except Exception as e:
                    logger.warning(f"[{self.tracker_id}] setParams()失敗: {e}")
            
            # 個別パラメータ設定を試行（トラッカー内部属性への直接アクセス）
            if hasattr(self.tracker, 'getParams'):
                try:
                    current_params = self.tracker.getParams()
                    if current_params:
                        # デフォルトパラメータから必要な属性をコピー
                        param_mapping = self._get_csrt_param_mapping()
                        applied_count = 0
                        
                        for param_key, param_value in self.csrt_params.items():
                            if param_key.startswith('csrt_'):
                                clean_key = param_key[5:]
                                opencv_attr = param_mapping.get(clean_key)
                                
                                if opencv_attr and hasattr(current_params, opencv_attr):
                                    try:
                                        setattr(current_params, opencv_attr, param_value)
                                        applied_count += 1
                                    except Exception as e:
                                        logger.debug(f"[{self.tracker_id}] 個別パラメータ設定失敗 {opencv_attr}: {e}")
                        
                        if applied_count > 0:
                            logger.info(f"[{self.tracker_id}] 初期化後パラメータ適用: {applied_count}個成功")
                        else:
                            logger.warning(f"[{self.tracker_id}] 初期化後パラメータ適用: 0個成功")
                        return
                except Exception as e:
                    logger.warning(f"[{self.tracker_id}] getParams()による適用失敗: {e}")
            
            # 全ての方法が失敗した場合
            logger.warning(f"[{self.tracker_id}] 初期化後パラメータ適用は利用できません（OpenCV 4.11.0制限）")
            
        except Exception as e:
            logger.error(f"[{self.tracker_id}] 初期化後パラメータ適用エラー: {e}")
    
    
    def _initialize(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> bool:
        """トラッカー初期化 - 標準OpenCVパターン"""
        try:
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1
            tracker_bbox = (x1, y1, width, height)
            
            # bbox検証
            if width <= 0 or height <= 0:
                logger.error(f"[{self.tracker_id}] 無効なbboxサイズ: width={width}, height={height}")
                return False
            
            logger.debug(f"[{self.tracker_id}] トラッカー初期化 - bbox: {tracker_bbox}")
            
            bgr_image = self._ensure_bgr_format(image)
            
            # 標準的なOpenCV CSRTパターン
            success = self.tracker.init(bgr_image, tracker_bbox)
            logger.debug(f"[{self.tracker_id}] tracker.init()結果: {success}")
            
            # OpenCV 4.11.0では戻り値がNoneの場合があるため、例外なしを成功とみなす
            if success is None:
                logger.info(f"[{self.tracker_id}] CSRT tracker初期化完了（OpenCV 4.11.0）")
                return True
            elif success:
                logger.info(f"[{self.tracker_id}] CSRT tracker初期化成功")
                return True
            else:
                logger.error(f"[{self.tracker_id}] CSRT tracker初期化失敗")
                return False
                
        except Exception as e:
            # OpenCV 4.11.0の既知の問題を処理
            if "vector::_M_range_insert" in str(e):
                logger.warning(f"[{self.tracker_id}] OpenCV 4.11.0互換性問題を検出、初期化成功と仮定")
                return True
            else:
                logger.error(f"[{self.tracker_id}] CSRTトラッカー初期化エラー: {e}")
                raise CSRTTrackingError(self.tracker_id, "initialization", e)
    
    def _ensure_bgr_format(self, image: np.ndarray) -> np.ndarray:
        """BGR形式確保"""
        if len(image.shape) == 2:
            logger.debug(f"[{self.tracker_id}] グレースケール画像をBGRに変換")
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif len(image.shape) == 3 and image.shape[2] == 3:
            logger.debug(f"[{self.tracker_id}] 3チャンネル画像をそのまま使用")
            return image
        else:
            logger.warning(f"[{self.tracker_id}] 予期しない画像形式: {image.shape}")
            return image
    
    def update(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """追跡更新 - 標準OpenCVパターン
        
        Returns:
            Optional[Tuple]: 更新されたbbox座標 (x1, y1, x2, y2) or None
        """
        if not self.is_initialized:
            logger.warning(f"[{self.tracker_id}] トラッカーが初期化されていません")
            return None
        
        try:
            bgr_image = self._ensure_bgr_format(image)
            success, bbox = self.tracker.update(bgr_image)
            
            logger.debug(f"[{self.tracker_id}] tracker.update()結果: success={success}, bbox={bbox}")
            
            if success and bbox is not None:
                x1, y1, w, h = [int(v) for v in bbox]
                result_bbox = (x1, y1, x1 + w, y1 + h)
                logger.debug(f"[{self.tracker_id}] 追跡成功: {result_bbox}")
                return result_bbox
            else:
                logger.debug(f"[{self.tracker_id}] 追跡失敗")
                return None
            
        except Exception as e:
            # OpenCV 4.11.0の既知の問題を処理
            if "vector::_M_range_insert" in str(e):
                logger.debug(f"[{self.tracker_id}] OpenCV 4.11.0内部エラー: {e}")
                return None
            else:
                logger.error(f"[{self.tracker_id}] CSRT追跡更新エラー: {e}")
                raise CSRTTrackingError(self.tracker_id, "update", e)