import Svg from './Svg';
import React from 'react';
import '../css/density.css';

const d3 = window.d3;

export default class DensityPlot extends React.Component {
    render() {
        return (
            <div>
                <Graph
                    topicA={this.props.topics[this.props.topicA].scores}
                    topicB={this.props.topics[this.props.topicB].scores}
                    colorA={this.props.topicA_color}
                    colorB={this.props.topicB_color}
                />
            </div>
        );
    }
}

const Graph = Svg((node, props) => {
    var n = 2,
        m = 25,
        stack = d3.stack().keys([0, 1]);

    stack.value(function (d, key) {
        return d[key].y;
    });
    var layers = stack(d3.transpose(d3.range(n).map(function (i) {
        return bumpLayer(i, m);
    })))
    // ,
    //    yStackMax = d3.max(layers, function (layer) { return d3.max(layer, function (d) { return d[1]; }); });

    var margin = { top: 40, right: 10, bottom: 20, left: 10 },
        width = (980 - margin.left - margin.right)/2 - 80,
        height = 300 - margin.top - margin.bottom //,
    //    x_axis_title_height = 20;

    var x = d3.scaleLinear()
        .domain([0, 7])
        .range([0, width]);

    var y = d3.scaleLinear()
        .domain([0, 0.35])
        .range([height, 0]);

    var xAxis = d3.axisBottom(x)
        .tickSize(10)
        .tickPadding(6);

    var yAxis = d3.axisLeft(y)
        .tickSize(10)
        .tickPadding(6);

    var svg = d3.select(node);
    svg.selectAll('*').remove();

    svg.attr("width", width + margin.left + margin.right + 80)
        .attr("height", height + margin.top + margin.bottom + 40)
        .style("padding", "40px 20px 20px 80px")
        .append("g")
            .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

    var rect = d3.area().x(function (d, i) { return x(d.data[0].x); })
        // .y0(function (d) { return y(d[0]); })
        .y0(height)
        .y1(function (d) { return y(d[1] - d[0]); })
        .curve(d3.curveBasis)

    svg.selectAll("path").data(layers).enter()
        .append("path")
            .attr("d", rect)
            .style("fill-opacity", 0.5)
            .style("fill", function (d, i) { return i ? props.colorA : props.colorB; })
            .attr("class", "area")
            .style("stroke", "black")
            .on("click", function () {
                // let paths = svg.selectAll(".area")._groups[0]
                // const d_0 = d3.select(paths[0]).attr('d')
                // const d_1 = d3.select(paths[1]).attr('d')

                // d3.select(paths[0]).attr('d', d_1)
                // d3.select(paths[1]).attr('d', d_0)
        })

    // add the x-axis
    svg.append("g")
        .attr("class", "x axis")
        .attr("transform", "translate(0," + height + ")")
        .call(xAxis);

    // text label for the x axis
    svg.append("text")             
        .attr("transform",
        "translate(" + (width/2) + " ," + 
                    (height + margin.top) + ")")
        .style("text-anchor", "middle")
        .text("Joke Score");

    // add the y-axis
    svg.append("g")
        .attr("class", "y axis")
        .attr("transform", "translate(0," + -0.5 + ")")
        .call(yAxis);

    // text label for the y axis
    svg.append("text")
        .attr("transform", "rotate(-90)")
        .attr("y", 0 - margin.left)
        .attr("x",0 - (height / 2))
        .attr("dy", "-2.5em")
        .style("text-anchor", "middle")
        .text("Topic Density");      

    // add a graph title
    svg.append("text")
        .attr("x", (width / 2))             
        .attr("y", 0 - (margin.top / 2))
        .attr("text-anchor", "middle")  
        .style("font-size", "16px")
        .text("Better topics have more area to the right");

    function bumpLayer(topic, bins) {
        const data = topic ? props.topicA : props.topicB;

        const res = new Array(bins).fill(0);
        let count = 0;
        const width = (bins / 11);

        let max = -1;
        for (let i = 0; i < data.length; i++) {
            max = max > data[i] ? max : data[i];
            const bin = Math.floor(data[i] * (width));
            res[bin]++;
            count++;
        }

        for (let i = 0; i < res.length; i++) {
            res[i] = {
                x: (11 / bins) * i,
                y: res[i] / count,
            };
        }
        return res;
    }
});


