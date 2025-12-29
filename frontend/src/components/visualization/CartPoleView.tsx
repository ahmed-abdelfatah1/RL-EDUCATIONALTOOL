import { useEffect, useState } from "react"
import type { CartPoleRenderState } from "../../types/environment"
import { Card } from "../ui/card"
import { Move, Gauge, TrendingUp, AlertCircle, CheckCircle2 } from "lucide-react"

interface CartPoleViewProps {
  state: CartPoleRenderState
}

export function CartPoleView({ state }: CartPoleViewProps) {
  const cartPosition = state.cart_position ?? 0
  const cartVelocity = state.cart_velocity ?? 0
  const poleAngle = state.pole_angle ?? 0
  const poleAngularVelocity = state.pole_angular_velocity ?? 0
  const actionTaken = state.action_taken
  const reward = state.reward ?? 0
  const done = state.done ?? false

  const [showActionIndicator, setShowActionIndicator] = useState(false)
  const [lastAction, setLastAction] = useState<number | null>(null)

  // Track action changes for visual feedback
  useEffect(() => {
    if (actionTaken !== null && actionTaken !== lastAction) {
      setShowActionIndicator(true)
      setLastAction(actionTaken)
      const timer = setTimeout(() => setShowActionIndicator(false), 400)
      return () => clearTimeout(timer)
    }
  }, [actionTaken, lastAction])

  // Calculate positions and danger levels
  const minPos = -2.4
  const maxPos = 2.4
  const ratio = (cartPosition - minPos) / (maxPos - minPos)
  const leftPercent = Math.min(100, Math.max(0, ratio * 100))

  const angleDeg = (poleAngle * 180) / Math.PI
  
  // Danger level calculation
  const getDangerLevel = (): 'safe' | 'warning' | 'danger' => {
    const angleWarning = Math.abs(poleAngle) > 0.15
    const angleDanger = Math.abs(poleAngle) > 0.18
    const posWarning = Math.abs(cartPosition) > 2.0
    const posDanger = Math.abs(cartPosition) > 2.2
    
    if (angleDanger || posDanger || done) return 'danger'
    if (angleWarning || posWarning) return 'warning'
    return 'safe'
  }

  const dangerLevel = getDangerLevel()
  const showDangerZones = dangerLevel !== 'safe'

  // Wheel spin speed based on velocity
  const getWheelSpinClass = () => {
    const absVel = Math.abs(cartVelocity)
    if (absVel > 1.5) return 'wheel-spin-fast'
    if (absVel > 0.5) return ''
    return 'wheel-spin-slow'
  }

  return (
    <Card className="border-border bg-card/50 backdrop-blur-sm">
      <div className="p-6 space-y-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Move className="h-5 w-5 text-blue-500" />
            <h3 className="text-lg font-semibold text-foreground">CartPole</h3>
          </div>
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            {done ? (
              <>
                <AlertCircle className="h-4 w-4 text-red-500" />
                <span className="text-red-600 font-medium">Fallen</span>
              </>
            ) : (
              <>
                <CheckCircle2 className="h-4 w-4 text-green-500" />
                <span className="text-green-600 font-medium">Balanced</span>
              </>
            )}
          </div>
        </div>

        {/* Enhanced Visualization */}
        <div className="cartpole-environment aspect-video rounded-xl border border-border p-4 flex items-center justify-center relative overflow-visible">
          <div className="w-full max-w-2xl h-full relative flex items-center justify-center">
            
            {/* Danger Zones */}
            {showDangerZones && (
              <>
                <div className="danger-zone-left" />
                <div className="danger-zone-right" />
              </>
            )}
            {dangerLevel === 'warning' && (
              <>
                <div className="warning-zone-left" />
                <div className="warning-zone-right" />
              </>
            )}

            {/* Track */}
            <div className="absolute top-1/2 left-8 right-8 h-3 cartpole-track">
              <div className="track-rail absolute top-1 left-0 right-0" />
              <div className="track-rail absolute bottom-1 left-0 right-0" />
            </div>

            {/* Position markers */}
            {[-2, -1, 0, 1, 2].map((mark) => {
              const markPos = ((mark + 2.4) / 4.8) * 100
              return (
                <div
                  key={mark}
                  className="absolute top-1/2 text-xs text-muted-foreground/70 font-mono"
                  style={{
                    left: `calc(8% + ${markPos}% * 0.84)`,
                    transform: "translate(-50%, 50%)",
                  }}
                >
                  {mark}
                </div>
              )
            })}

            {/* Cart and Pole Container */}
            <div
              className="cartpole-cart"
              style={{
                left: `calc(8% + ${leftPercent}% * 0.84)`,
                transform: "translateX(-50%)",
                bottom: "calc(50% - 6px)",
              }}
            >
              {/* Cart Shadow */}
              <div
                className="cart-shadow"
                style={{
                  width: "80px",
                  height: "12px",
                  bottom: "-18px",
                  left: "50%",
                    transform: "translateX(-50%)",
                  }}
                />

              {/* Action Indicators */}
              {showActionIndicator && actionTaken !== null && (
                <div className={`action-indicator ${actionTaken === 0 ? 'action-indicator-left' : 'action-indicator-right'}`}>
                  {actionTaken === 0 ? '←' : '→'}
                </div>
              )}

              {/* Pole */}
              <div
                className={`cartpole-pole ${dangerLevel === 'safe' ? 'status-glow-safe' : dangerLevel === 'warning' ? 'status-glow-warning' : 'status-glow-danger'}`}
                style={{
                  transform: `translateX(-50%) rotate(${-angleDeg}deg)`,
                  width: "8px",
                  height: "100px",
                }}
              >
                {/* Pole Segments */}
                <div className="pole-segment" style={{ width: "8px", height: "30px", bottom: "38px" }} />
                <div className="pole-segment" style={{ width: "8px", height: "30px", bottom: "8px" }} />
                <div className="pole-segment" style={{ width: "8px", height: "30px", bottom: "68px" }} />
                
                {/* Pole Weight at top */}
                <div className="pole-weight" style={{ width: "20px", height: "20px" }} />
              </div>

              {/* Pole Joint */}
              <div className="pole-joint" style={{ width: "16px", height: "16px" }} />

              {/* Cart Body */}
              <div className="cart-body" style={{ width: "60px", height: "32px", position: "relative" }}>
                <div className="cart-top-panel" />
                
                {/* Suspension springs */}
                <div className="cart-suspension" style={{ left: "10px", bottom: "-8px", height: "8px" }} />
                <div className="cart-suspension" style={{ right: "10px", bottom: "-8px", height: "8px" }} />
              </div>

              {/* Wheels */}
              <div className={`cart-wheel ${getWheelSpinClass()}`} style={{ 
                width: "18px", 
                height: "18px", 
                left: "8px", 
                bottom: "-20px" 
              }}>
                <div className="wheel-spokes">
                  <div className="wheel-spoke" style={{ transform: "rotate(0deg)" }} />
                  <div className="wheel-spoke" style={{ transform: "rotate(45deg)" }} />
                  <div className="wheel-spoke" style={{ transform: "rotate(90deg)" }} />
                  <div className="wheel-spoke" style={{ transform: "rotate(135deg)" }} />
                </div>
              </div>
              <div className={`cart-wheel ${getWheelSpinClass()}`} style={{ 
                width: "18px", 
                height: "18px", 
                right: "8px", 
                bottom: "-20px" 
              }}>
                <div className="wheel-spokes">
                  <div className="wheel-spoke" style={{ transform: "rotate(0deg)" }} />
                  <div className="wheel-spoke" style={{ transform: "rotate(45deg)" }} />
                  <div className="wheel-spoke" style={{ transform: "rotate(90deg)" }} />
                  <div className="wheel-spoke" style={{ transform: "rotate(135deg)" }} />
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Stats Grid */}
        <div className="grid grid-cols-2 gap-4">
          <Card className="p-4 border-border bg-muted/30">
            <div className="flex items-center gap-2 mb-2">
              <Move className="h-4 w-4 text-muted-foreground" />
              <p className="text-xs text-muted-foreground">Cart Position</p>
            </div>
            <p className="text-lg font-semibold text-foreground">{cartPosition.toFixed(3)}</p>
          </Card>

          <Card className="p-4 border-border bg-muted/30">
            <div className="flex items-center gap-2 mb-2">
              <Gauge className="h-4 w-4 text-muted-foreground" />
              <p className="text-xs text-muted-foreground">Cart Velocity</p>
            </div>
            <p className="text-lg font-semibold text-foreground">{cartVelocity.toFixed(3)}</p>
          </Card>

          <Card className="p-4 border-border bg-muted/30">
            <div className="flex items-center gap-2 mb-2">
              <TrendingUp className="h-4 w-4 text-muted-foreground" />
              <p className="text-xs text-muted-foreground">Pole Angle</p>
            </div>
            <p className="text-lg font-semibold text-foreground">
              {poleAngle.toFixed(3)} rad ({angleDeg.toFixed(1)}°)
            </p>
          </Card>

          <Card className="p-4 border-border bg-muted/30">
            <div className="flex items-center gap-2 mb-2">
              <Gauge className="h-4 w-4 text-muted-foreground" />
              <p className="text-xs text-muted-foreground">Pole Velocity</p>
            </div>
            <p className="text-lg font-semibold text-foreground">{poleAngularVelocity.toFixed(3)}</p>
          </Card>

          <Card className="p-4 border-border bg-muted/30">
            <div className="flex items-center gap-2 mb-2">
              <TrendingUp className="h-4 w-4 text-muted-foreground" />
              <p className="text-xs text-muted-foreground">Reward</p>
            </div>
            <p className="text-lg font-semibold text-foreground">{reward.toFixed(1)}</p>
          </Card>

          <Card className={`p-4 border-border ${
            dangerLevel === 'danger' ? 'bg-red-500/10' : 
            dangerLevel === 'warning' ? 'bg-yellow-500/10' : 
            'bg-green-500/10'
          }`}>
            <div className="flex items-center gap-2 mb-2">
              {done ? (
                <AlertCircle className="h-4 w-4 text-red-500" />
              ) : dangerLevel === 'danger' ? (
                <AlertCircle className="h-4 w-4 text-red-500" />
              ) : dangerLevel === 'warning' ? (
                <AlertCircle className="h-4 w-4 text-yellow-500" />
              ) : (
                <CheckCircle2 className="h-4 w-4 text-green-500" />
              )}
              <p className="text-xs text-muted-foreground">Status</p>
            </div>
            <p className={`text-lg font-semibold ${
              done ? "text-red-600" : 
              dangerLevel === 'danger' ? "text-red-600" :
              dangerLevel === 'warning' ? "text-yellow-600" :
              "text-green-600"
            }`}>
              {done ? "Fallen! ✗" : 
               dangerLevel === 'danger' ? "Critical!" :
               dangerLevel === 'warning' ? "Warning!" :
               "Balanced ✓"}
            </p>
          </Card>
        </div>
      </div>
    </Card>
  )
}
