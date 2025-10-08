import React, { useRef, useEffect, useState, useCallback } from 'react';
import { Camera } from '@mediapipe/camera_utils';
import { Pose } from '@mediapipe/pose';

// MediaPipeの描画ユーティリティは、index.htmlでグローバルにロードされている前提
// window.drawConnectors, window.drawLandmarks が利用可能と仮定

// challenges.js からのインポート (パスは環境に合わせて調整してください)
import { CHALLENGES, TARGET_ANGLES, TOLERANCE, LANDMARKS as LANDMARK_INDICES } from '../data/challenges'; 

// =========================================================================
// 🎯 ヘルパー関数: ポーズ検出とスコア計算 (コンポーネント外部)
// =========================================================================

/**
 * 3つのランドマークから角度を計算
 */
const calculateAngle = (A, M, B) => {
    const vectorMA_x = A.x - M.x;
    const vectorMA_y = A.y - M.y;
    const vectorMB_x = B.x - M.x;
    const vectorMB_y = B.y - M.y;

    const dotProduct = (vectorMA_x * vectorMB_x) + (vectorMA_y * vectorMB_y);
    const magnitudeMA = Math.sqrt(Math.pow(vectorMA_x, 2) + Math.pow(vectorMA_y, 2));
    const magnitudeMB = Math.sqrt(Math.pow(vectorMB_x, 2) + Math.pow(vectorMB_y, 2));
    
    let angleDeg = 0;
    if (magnitudeMA !== 0 && magnitudeMB !== 0) {
        let cosTheta = dotProduct / (magnitudeMA * magnitudeMB);
        cosTheta = Math.max(-1, Math.min(1, cosTheta));
        let angleRad = Math.acos(cosTheta);
        angleDeg = angleRad * (180 / Math.PI);
    }
    return angleDeg;
};

/**
 * 起動トリガー用のポーズが取られているか判定する (両腕垂直上げ)
 */
const isStartPoseAchieved = (landmarks) => {
    const L = LANDMARK_INDICES;
    const target = TARGET_ANGLES;
    const tolerance = TOLERANCE.START_TOLERANCE;

    if (!landmarks) return false;

    const requiredLandmarks = [L.LEFT_SHOULDER, L.LEFT_ELBOW, L.LEFT_WRIST, L.LEFT_HIP,
                               L.RIGHT_SHOULDER, L.RIGHT_ELBOW, L.RIGHT_WRIST, L.RIGHT_HIP];
    for (const index of requiredLandmarks) {
        if (!landmarks[index] || landmarks[index].visibility < 0.7) {
            return false;
        }
    }

    const leftElbowAngle = calculateAngle(landmarks[L.LEFT_SHOULDER], landmarks[L.LEFT_ELBOW], landmarks[L.LEFT_WRIST]);
    const leftShoulderAngle = calculateAngle(landmarks[L.LEFT_HIP], landmarks[L.LEFT_SHOULDER], landmarks[L.LEFT_ELBOW]);
    const isLeftArmReady = (
        Math.abs(leftElbowAngle - target.ELBOW) <= tolerance &&
        Math.abs(leftShoulderAngle - target.SHOULDER) <= tolerance
    );

    const rightElbowAngle = calculateAngle(landmarks[L.RIGHT_SHOULDER], landmarks[L.RIGHT_ELBOW], landmarks[L.RIGHT_WRIST]);
    const rightShoulderAngle = calculateAngle(landmarks[L.RIGHT_HIP], landmarks[L.RIGHT_SHOULDER], landmarks[L.RIGHT_ELBOW]);
    const isRightArmReady = (
        Math.abs(rightElbowAngle - target.ELBOW) <= tolerance &&
        Math.abs(rightShoulderAngle - target.SHOULDER) <= tolerance
    );

    return isLeftArmReady && isRightArmReady;
};


// =========================================================================
// 🚀 Reactコンポーネント
// =========================================================================

const PoseChallenge = () => {
    // DOM参照
    const videoRef = useRef(null);
    const canvasRef = useRef(null);
    const poseRef = useRef(null); // Poseインスタンスを保持

    // 状態管理
    const [currentChallengeIndex, setCurrentChallengeIndex] = useState(0);
    const [challengeState, setChallengeState] = useState({
        isPoseFixed: false,
        isChallengeStarted: false,
        isInPreparationPhase: false,
        finalPoseLandmarks: null,
        currentScore: '--',
        guideMessage: 'カメラ起動中...',
        timerDisplay: null,
        challenges: JSON.parse(JSON.stringify(CHALLENGES)), // チャレンジリストを状態としてコピー
        scoreColor: '#4CAF50',
    });
    
    // スコア計算ロジック (useCallbackでメモ化)
    const calculateMatchScore = useCallback((currentLandmarks) => {
        const challenge = challengeState.challenges[currentChallengeIndex];
        const L = LANDMARK_INDICES;
        const target = TARGET_ANGLES;
        const tolerance = TOLERANCE;
        let totalScore = 0;
        let jointCount = 0;

        if (!challenge) return 0;

        // --- 1. ARMチャレンジ (上半身) の評価 ---
        if (challenge.targetType === 'ARM') {
            const side = challenge.evalJoints[0].includes('L') ? 'LEFT' : 'RIGHT';
            
            const S = L[`${side}_SHOULDER`];
            const E = L[`${side}_ELBOW`];
            const W = L[`${side}_WRIST`];
            const H = L[`${side}_HIP`];

            if (currentLandmarks[S].visibility < 0.7 || currentLandmarks[E].visibility < 0.7 || 
                currentLandmarks[W].visibility < 0.7 || currentLandmarks[H].visibility < 0.7) {
                return 0;
            }

            const currentElbowAngle = calculateAngle(currentLandmarks[S], currentLandmarks[E], currentLandmarks[W]);
            const currentShoulderAngle = calculateAngle(currentLandmarks[H], currentLandmarks[S], currentLandmarks[E]);

            const diffElbow = Math.abs(currentElbowAngle - target.ELBOW);
            const diffShoulder = Math.abs(currentShoulderAngle - target.SHOULDER);
            
            const scoreElbow = 100 * (1 - (diffElbow / tolerance.ELBOW));
            const scoreShoulder = 100 * (1 - (diffShoulder / tolerance.SHOULDER));

            totalScore += Math.max(0, scoreElbow);
            totalScore += Math.max(0, scoreShoulder);
            jointCount += 2;
        } 
        
        // --- 2. LEG_BALANCEチャレンジ (全身) の評価 ---
        else if (challenge.targetType === 'LEG_BALANCE') {
            // 右足軸の片足立ちを想定
            const R_K = L.RIGHT_KNEE;
            const R_A = L.RIGHT_ANKLE;
            const R_H = L.RIGHT_HIP;
            const L_H = L.LEFT_HIP;

            // 可視性チェック: 軸足の膝・足首と両腰
            if (currentLandmarks[R_K].visibility < 0.7 || currentLandmarks[R_A].visibility < 0.7 || 
                currentLandmarks[R_H].visibility < 0.7 || currentLandmarks[L_H].visibility < 0.7) {
                return 0;
            }

            // 2-1. 軸足の膝の伸び (腰-膝-足首)
            const currentKneeAngle = calculateAngle(currentLandmarks[R_H], currentLandmarks[R_K], currentLandmarks[R_A]);
            const diffKnee = Math.abs(currentKneeAngle - target.KNEE_STRAIGHT);
            const scoreKnee = 100 * (1 - (diffKnee / tolerance.KNEE));
            
            totalScore += Math.max(0, scoreKnee);
            jointCount += 1;

            // 2-2. 体幹の垂直安定性 (軸足のヒップと膝を結ぶ線の傾き)
            const angleRad = Math.atan2(currentLandmarks[R_H].x - currentLandmarks[R_K].x, currentLandmarks[R_H].y - currentLandmarks[R_K].y);
            const verticalAngle = Math.abs(angleRad * (180 / Math.PI)); 
            const tiltDiff = Math.min(verticalAngle, Math.abs(180 - verticalAngle)); 
            
            const scoreTilt = 100 * (1 - (tiltDiff / tolerance.TILT));
            
            totalScore += Math.max(0, scoreTilt);
            jointCount += 1;
            
            // 2-3. 腰の水平バランスチェック（左右の腰のY座標の差が少ないほど良い）
            const canvasElement = canvasRef.current;
            const hipDeltaY = Math.abs(currentLandmarks[R_H].y - currentLandmarks[L_H].y) * (canvasElement ? canvasElement.height : 480); 
            
            // 20ピクセル以内のズレを100点として評価
            const hipBalanceScore = Math.max(0, 100 * (1 - (hipDeltaY / 20))); 
            
            totalScore += hipBalanceScore;
            jointCount += 1;
        }

        if (jointCount === 0) return 0;
        
        return parseFloat((totalScore / jointCount).toFixed(1)); 
    }, [challengeState.challenges, currentChallengeIndex]);


    // --- チャレンジ管理関数 ---

    const showFinalResults = useCallback(() => {
        let totalScore = 0;
        let scoreList = "";
        
        challengeState.challenges.forEach((c) => {
            totalScore += c.score;
            scoreList += `${c.name}: ${c.score.toFixed(1)}%\n`;
        });
        
        const averageScore = totalScore / challengeState.challenges.length;
        
        let message, scoreColor;
        if (averageScore > 90) {
            scoreColor = '#4CAF50';
            message = `🎉 **全チャレンジ完了！** 平均スコア: ${averageScore.toFixed(1)}%<br>素晴らしいパーフェクト達成です！`;
        } else {
            scoreColor = '#FFC107';
            message = `**全チャレンジ完了！** 平均スコア: ${averageScore.toFixed(1)}%<br>結果をコンソールで確認し、もう一度チャレンジしましょう！`;
        }

        setChallengeState(prev => ({
            ...prev,
            currentScore: `${averageScore.toFixed(1)} % (平均)`,
            guideMessage: message,
            scoreColor: scoreColor
        }));

        console.log("--- FINAL CHALLENGE RESULTS ---");
        console.log(scoreList);
        console.log(`Average Score: ${averageScore.toFixed(1)}%`);
    }, [challengeState.challenges]);


    const resetChallenge = useCallback((nextStage = false) => {
        let newIndex = currentChallengeIndex;
        if (nextStage) {
            newIndex += 1;
        }

        if (newIndex < challengeState.challenges.length) {
            setChallengeState(prev => ({
                ...prev,
                isPoseFixed: false,
                finalPoseLandmarks: null,
                isChallengeStarted: false,
                isInPreparationPhase: false,
                currentScore: '--',
                guideMessage: `チャレンジ開始のため、両手を垂直に上げてポーズを維持してください。`,
                timerDisplay: null,
                scoreColor: '#4CAF50',
            }));
            setCurrentChallengeIndex(newIndex);
        } else {
            showFinalResults();
        }
    }, [currentChallengeIndex, challengeState.challenges, showFinalResults]);


    const startChallengeTimer = useCallback(() => {
        if (challengeState.isChallengeStarted) return;
        
        setChallengeState(prev => ({ ...prev, isChallengeStarted: true }));

        const COUNTDOWN_SECONDS = 3;
        const HOLD_SECONDS = 5;
        // const TOTAL_DELAY_SECONDS = COUNTDOWN_SECONDS + HOLD_SECONDS; // ローカルでの参照は不要

        let startTime = Date.now();

        const timerId = setInterval(() => {
            const elapsed = Math.floor((Date.now() - startTime) / 1000);
            
            if (elapsed < COUNTDOWN_SECONDS) {
                const remaining = COUNTDOWN_SECONDS - elapsed;
                setChallengeState(prev => ({ 
                    ...prev, 
                    timerDisplay: remaining, 
                    guideMessage: `ポーズを取る準備！残り ${remaining} 秒！` 
                }));
            } else if (elapsed < (COUNTDOWN_SECONDS + HOLD_SECONDS)) {
                const holdTime = elapsed - COUNTDOWN_SECONDS;
                setChallengeState(prev => ({ 
                    ...prev, 
                    timerDisplay: 'GO!', 
                    guideMessage: `ポーズを維持してください！測定中... ${holdTime + 1} / ${HOLD_SECONDS} 秒` 
                }));
            } else {
                clearInterval(timerId);
                setChallengeState(prev => ({ 
                    ...prev, 
                    isPoseFixed: true, 
                    timerDisplay: null,
                    guideMessage: 'ポーズ確定！最終スコアを計算中です。'
                }));
            }
        }, 1000);

        return () => clearInterval(timerId);
    }, [challengeState.isChallengeStarted]);


    const startPreparationPhase = useCallback(() => {
        if (challengeState.isChallengeStarted || challengeState.isInPreparationPhase) return;
        
        const currentChallenge = challengeState.challenges[currentChallengeIndex];
        
        setChallengeState(prev => ({
            ...prev,
            isInPreparationPhase: true,
            guideMessage: `✅ ${currentChallenge.message} ポーズを確認！そのままでお待ちください...`
        }));

        setTimeout(() => {
            setChallengeState(prev => ({ ...prev, isInPreparationPhase: false }));
            startChallengeTimer();
        }, 1500); // 1.5秒の準備期間
    }, [challengeState, currentChallengeIndex, startChallengeTimer]);


    // --- MediaPipe 結果処理 ---

    const onResults = useCallback((results) => {
        const canvasElement = canvasRef.current;
        if (!canvasElement) return;

        const canvasCtx = canvasElement.getContext('2d');

        canvasCtx.save();
        canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
        canvasCtx.globalCompositeOperation = 'source_over';
        canvasCtx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);

        if (results.poseLandmarks) {
            
            // 1. チャレンジ開始チェック
            if (!challengeState.isChallengeStarted && !challengeState.isInPreparationPhase && currentChallengeIndex < challengeState.challenges.length && isStartPoseAchieved(results.poseLandmarks)) {
                startPreparationPhase();
            }

            // 2. ポーズ確定の瞬間、データを保存
            if (challengeState.isPoseFixed && !challengeState.finalPoseLandmarks) {
                const finalLandmarks = JSON.parse(JSON.stringify(results.poseLandmarks));
                const score = calculateMatchScore(finalLandmarks);
                
                // 状態を更新
                setChallengeState(prev => {
                    const updatedChallenges = [...prev.challenges];
                    updatedChallenges[currentChallengeIndex].score = score;
                    
                    let message, scoreColor;
                    if (score > 90) {
                        scoreColor = '#4CAF50'; 
                        message = `🌟 ${prev.challenges[currentChallengeIndex].name} 完了！パーフェクト達成です！`;
                    } else if (score > 70) {
                        scoreColor = '#FFC107';
                        message = `${prev.challenges[currentChallengeIndex].name} 完了！もう少しで目標達成でした！`;
                    } else {
                        scoreColor = '#F44336';
                        message = `${prev.challenges[currentChallengeIndex].name} 完了。惜しかったです！`;
                    }

                    return {
                        ...prev,
                        finalPoseLandmarks: finalLandmarks,
                        challenges: updatedChallenges,
                        currentScore: `${score.toFixed(1)} % (FINAL)`,
                        guideMessage: message,
                        scoreColor: scoreColor
                    };
                });

                // 1秒後に次のチャレンジへ移行または終了
                setTimeout(() => {
                    resetChallenge(true);
                }, 1000);
            }

            // 3. 描画とリアルタイムスコアの更新
            const drawingLandmarks = challengeState.finalPoseLandmarks || results.poseLandmarks;
            const isFixed = challengeState.isPoseFixed;

            const lineColor = isFixed ? '#FFD700' : '#00FF00'; 
            const dotColor = isFixed ? '#FFA500' : '#FF0000'; 
            
            if (window.drawConnectors && window.drawLandmarks) {
                window.drawConnectors(canvasCtx, drawingLandmarks, window.POSE_CONNECTIONS,
                                   { color: lineColor, lineWidth: 4 }); 
                window.drawLandmarks(canvasCtx, drawingLandmarks,
                                    { color: dotColor, lineWidth: 2, radius: 4 });
            }

            // リアルタイムスコアの更新
            if (!challengeState.finalPoseLandmarks && challengeState.isChallengeStarted) {
                const score = calculateMatchScore(results.poseLandmarks);
                let scoreColor;
                if (score > 80) scoreColor = '#4CAF50';
                else if (score > 50) scoreColor = '#FFA500';
                else scoreColor = '#F44336';

                setChallengeState(prev => ({
                    ...prev,
                    currentScore: `${score.toFixed(1)} %`,
                    scoreColor: scoreColor
                }));
            }
        }
        
        canvasCtx.restore();
    }, [challengeState, currentChallengeIndex, calculateMatchScore, startPreparationPhase, resetChallenge]);


    // --- MediaPipe & カメラ初期化 (useEffect) ---

    useEffect(() => {
        const videoElement = videoRef.current;
        
        // Poseインスタンスがなければ作成
        if (!poseRef.current) {
             poseRef.current = new Pose({
                locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`,
            });

            poseRef.current.setOptions({
                modelComplexity: 1, 
                smoothLandmarks: true
            });
            poseRef.current.onResults(onResults);
        }

        const camera = new Camera(videoElement, {
            onFrame: async () => {
                // isPoseFixed が false の場合のみ検出フレームを送信
                if (!challengeState.isPoseFixed && poseRef.current) {
                    await poseRef.current.send({ image: videoElement });
                }
                // 固定されている場合、onResultsはuseEffect外のロジックで手動で呼ばれる
            },
            width: 640,
            height: 480
        });

        camera.start()
            .then(() => {
                setChallengeState(prev => ({
                    ...prev,
                    guideMessage: `カメラ起動完了！チャレンジ開始のため、両手を垂直に上げてポーズを維持してください。`
                }));
            })
            .catch(error => {
                setChallengeState(prev => ({
                    ...prev,
                    guideMessage: `エラー: カメラの起動に失敗しました。アクセスを許可してください。 (${error.name})`
                }));
                console.error("Camera start failed:", error);
            });

        // クリーンアップ
        return () => {
            // Cameraインスタンスがクリーンアップされることを確認
            // MediaPipeのインスタンス自体はコンポーネントがアンマウントされるまで保持
        };
    }, [onResults, challengeState.isPoseFixed]); // challengeState.finalPoseLandmarks は不要 (onResultsが依存しているため)


    // --- JSX レンダリング ---

    return (
        <div id="main-layout">
            <div id="video-container">
                {/* MediaPipeの内部処理に使うビデオ要素 */}
                <video ref={videoRef} id="video" playsInline></video>
                {/* 描画結果を表示するキャンバス要素 */}
                <canvas ref={canvasRef} id="canvas"></canvas>
            </div>

            <div id="result-box">
                <div 
                    id="timer-display" 
                    className={challengeState.timerDisplay !== null ? 'show-timer' : ''}
                >
                    {challengeState.timerDisplay}
                </div>
                
                <p>
                    現在のマッチング度: 
                    <span 
                        id="match-score" 
                        style={{ color: challengeState.scoreColor || '#4CAF50' }}
                    >
                        {challengeState.currentScore}
                    </span>
                </p>
                {/* dangerouslySetInnerHTMLを使用してHTMLタグを含むメッセージを表示 */}
                <div id="guide-message" dangerouslySetInnerHTML={{ __html: challengeState.guideMessage }}>
                </div>
            </div>
        </div>
    );
};

export default PoseChallenge;