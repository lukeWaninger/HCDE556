import Svg from './Svg';
import React from 'react';
import '../css/bar_charts.css';

const d3 = window.d3;

export default class Bar extends React.Component {
    getBarData() { 
        const start = this.props.word_page * this.props.words_per_bar;

        let set = (this.props.topic === 1)
            ? this.props.topics[this.props.topicA].words.slice(start, start+this.props.words_per_bar)
            : this.props.topics[this.props.topicB].words.slice(start, start+this.props.words_per_bar)
        set = set.map(v => {
            return {
                p_topic: v.prob,
                count: v.count,
                name: v.name,
                topic_dist: v.topic_dist
            }
        });
        return set;
    }

    getMaxProb() {
        return (this.props.topic === 1)
            ? d3.max(this.props.topics[this.props.topicA].words.map(x => {
                return x.prob
            }))
            : d3.max(this.props.topics[this.props.topicB].words.map(x => {
                return x.prob
            }))
    }

    getColor() {
        const tColor = (this.props.topic === 1)
            ? this.props.topicA_color
            : this.props.topicB_color
        return tColor;
    }

    render() {
        return (
            <div className="bar-container">
                <BarChart
                    data={this.getBarData()}
                    maxProb={this.getMaxProb()}
                    color={this.getColor()}
                    onWordSelect={this.props.onSelect}
                />
            </div>
        );
    }
}

const BarChart = Svg((node, props) => {   
    // chart dimensions
    var width = 300,
        height = 440,
        barWidth = 20,
        barSpacing = 1;

    const countMax = d3.max(props.data.map(v => {
        return v.count
    }));

    function prob(d) { return d.p_topic; }
    function w_count(d) { return d.count; }

    const lengthScale = d3.scaleLinear()
        .domain([0, props.maxProb])
        .range([5, width - 80])

    const colorScale = d3.scaleLinear()
        .domain([0, countMax])
        .range(['#444', props.color])

    var svg = d3.select(node);
    svg.selectAll('*').remove();

    svg.attr('height', height)

    // add words
    svg.selectAll('g')
        .data(props.data)
        .enter()
        .append('text')
        .attr('y', (d, i) => i * (barWidth + barSpacing))
        .attr('dy', '1em')
        .attr('x', d => 0)
        .attr('class', 'bar-title')
        .text(function (d) { return d.name; })

    // add bars
    svg.selectAll('rect')
        .data(props.data)
        .enter()
        .append('rect')
        .attr('y', (d, i) => i * (barWidth + barSpacing))
        .attr('x', d => 100)
        .attr('width', d => lengthScale(prob(d)))
        .attr('height', barWidth)
        .style('fill', d => colorScale(d.count))
        .on('click', (d, i, nodes) => props.onWordSelect(d, i))
        .on('mouseover', (data, index, nodes) => {
            d3.select(nodes[index])
                .style('fill', '#444')

            // create little histogram           
            var lhmargin = { top: 10, right: 5, bottom: 5, left: 5},
                lhwidth = 200 - lhmargin.left - lhmargin.right,
                lheight = 200,
                lhbar = 5,
                lhspacing = 0

            var lilHisty = document.createElement("svg");          
            lilHisty.id = "lilhisty"

            const lhMax = d3.max(data.topic_dist);        

            // position scales
            const lhYScale = d3.scaleLinear()
                .domain([0, lhMax])
                .range([0, lheight])      

            // build the histy boi
            lilHisty = d3.select(lilHisty);
            lilHisty.selectAll('*').remove();
        
            lilHisty.attr("width", lhwidth + lhmargin.left + lhmargin.right)
                .attr("height", lhwidth + lhmargin.top + lhmargin.bottom + 100)
        
            // add bars
            lilHisty.selectAll('rect')
                .data(data.topic_dist)
                .enter()
                .append('rect')
                    .attr('x', (d, i) => i * (lhbar + lhspacing))
                    .attr('height', d => lhYScale(d))
                    .attr('y', d => lheight - lhYScale(d) - 100)
                    .attr('width', lhbar)
                    .style('fill', (d, i) => props.color)            

            d3.select("#tooltip")
                .html(
                "<h4>" + data.name + "</h4>" +
                "<p><em>affinity:</em> " + data.p_topic.toFixed(5) +
                "</br><em>count:</em> " + data.count +
                "</br><em>affinity to all topics</em>" +
                "</p><svg width:" + lhwidth + ";height:" + lheight + ">" +  lilHisty.html() + "</svg>"
                )
            
        })
        .on('mouseout', function (data, index, nodes) {
            let col = colorScale(w_count(data))

            let xPos = d3.event.clientX + 10;
            let yPos = d3.event.clientY + 10;
            if (xPos > 900) {
                xPos -= 90;
                yPos += 30;
            }            

            d3.select(nodes[index])
                .transition()
                .duration(200)
                .style('fill', col);

            d3.select("#tooltip")
                .attr('style',
                'opacity:0;border: 1px solid ' + col +
                ';border-top: 15px solid ' + col +
                ';top:' + yPos +
                'px;left:' + xPos + "px")
        })
        .on('mousemove', function (data, index, nodes) {
            let col = colorScale(w_count(data))

            let xPos = d3.event.clientX + 10;
            let yPos = d3.event.clientY + 10;
            if (xPos > 900) {
                xPos -= 90;
                yPos += 30;
            }  

            d3.select("#tooltip")
                .attr('style',
                'opacity:.95;border: 1px solid ' + col +
                ';border-top: 15px solid ' + col +
                ';top:' + yPos +
                'px;left:' + xPos + "px")
        })
});