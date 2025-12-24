import { useMemo } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "../ui/card"
import { Activity } from "lucide-react"

interface ValueFunctionViewProps {
    envName: string
    valueFunction: Record<string, number>
}

export function ValueFunctionView({ envName, valueFunction }: ValueFunctionViewProps) {
    const gridSize = useMemo(() => {
        if (envName === "gridworld") return 5
        if (envName === "frozenlake") return 4
        return 0
    }, [envName])

    const { minVal, maxVal } = useMemo(() => {
        const values = Object.values(valueFunction)
        if (values.length === 0) return { minVal: 0, maxVal: 1 }
        return {
            minVal: Math.min(...values),
            maxVal: Math.max(...values),
        }
    }, [valueFunction])

    const getColor = (value: number) => {
        if (maxVal === minVal) return "rgb(200, 200, 200)"

        // Normalize to 0-1
        const normalized = (value - minVal) / (maxVal - minVal)

        // Red (low) to Green (high)
        // Low: 255, 0, 0
        // Mid: 255, 255, 0
        // High: 0, 255, 0

        let r, g, b
        if (normalized < 0.5) {
            // Red to Yellow
            r = 255
            g = Math.round(normalized * 2 * 255)
            b = 0
        } else {
            // Yellow to Green
            r = Math.round((1 - normalized) * 2 * 255)
            g = 255
            b = 0
        }

        return `rgba(${r}, ${g}, ${b}, 0.8)`
    }

    if (gridSize === 0) {
        return (
            <Card className="border-border bg-card/50 backdrop-blur-sm h-full">
                <CardHeader>
                    <CardTitle className="text-sm text-muted-foreground">
                        Value function visualization not available for this environment.
                    </CardTitle>
                </CardHeader>
            </Card>
        )
    }

    const cells = Array.from({ length: gridSize * gridSize }, (_, i) => i)

    return (
        <Card className="border-border bg-card/50 backdrop-blur-sm h-full">
            <CardHeader className="pb-2">
                <div className="flex items-center gap-2">
                    <Activity className="h-5 w-5 text-green-500" />
                    <CardTitle className="text-lg">Value Function Heatmap</CardTitle>
                </div>
            </CardHeader>
            <CardContent>
                <div
                    className="grid gap-1 w-full aspect-square max-w-[300px] mx-auto"
                    style={{
                        gridTemplateColumns: `repeat(${gridSize}, 1fr)`,
                    }}
                >
                    {cells.map((stateId) => {
                        const value = valueFunction[stateId.toString()] ?? 0
                        return (
                            <div
                                key={stateId}
                                className="relative flex items-center justify-center border border-border/50 rounded-sm text-xs font-mono transition-colors hover:border-foreground/50"
                                style={{ backgroundColor: getColor(value) }}
                                title={`State ${stateId}: ${value.toFixed(3)}`}
                            >
                                <span className="bg-background/80 px-1 rounded backdrop-blur-[1px]">
                                    {value.toFixed(1)}
                                </span>
                            </div>
                        )
                    })}
                </div>
                <div className="flex justify-between text-xs text-muted-foreground mt-4 px-4">
                    <span>Min: {minVal.toFixed(2)}</span>
                    <div className="flex gap-1 items-center">
                        <div className="w-16 h-2 bg-gradient-to-r from-red-500 via-yellow-500 to-green-500 rounded-full" />
                    </div>
                    <span>Max: {maxVal.toFixed(2)}</span>
                </div>
            </CardContent>
        </Card>
    )
}
